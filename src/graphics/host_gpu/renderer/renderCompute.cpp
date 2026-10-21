#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuCrashDiag.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>
#include <optional>
#include <tuple>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

static void FillDiag(uint64_t shader, int outcome);

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					FillDiag(program.shader_hash, 5);
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

// A proven fill kept as a native dispatch (its memory is shared with cached images): recorded in
// g_gpu_fills after the dispatch, whose own write marking would erase an earlier record.
static std::optional<std::tuple<uint64_t, uint64_t, uint32_t>> g_pending_native_fill;

// Diagnostic (uncommitted): compute fill outcomes per shader, printed every 512 fill candidates.
static void FillDiag(uint64_t shader, int outcome) {
	static std::map<uint64_t, std::array<uint64_t, 8>> counts;
	static uint64_t                                    total = 0;
	counts[shader][outcome]++;
	if (++total % 512 != 0) {
		return;
	}
	for (const auto& [hash, c]: counts) {
		std::fprintf(stderr,
		             "[filldiag] cs %016llx: not resolved %llu, image clear %llu, kept native "
		             "(image memory) %llu, CPU fill %llu, unaligned/unmapped %llu, meta clear consumed "
		             "%llu, skipped (specialization reads) %llu, indirect %llu\n",
		             static_cast<unsigned long long>(hash), static_cast<unsigned long long>(c[0]),
		             static_cast<unsigned long long>(c[1]), static_cast<unsigned long long>(c[2]),
		             static_cast<unsigned long long>(c[3]), static_cast<unsigned long long>(c[4]),
		             static_cast<unsigned long long>(c[5]), static_cast<unsigned long long>(c[6]),
		             static_cast<unsigned long long>(c[7]));
	}
	std::fflush(stderr);
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	const bool diag_candidate = fill.kind == ShaderRecompiler::IR::UniformFillKind::Buffer;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		if (diag_candidate) FillDiag(program.shader_hash, 0);
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size)) {
			FillDiag(program.shader_hash, 4);
			return false;
		}
		if (cache.IsRegionRegistered(descriptor.Base48(), size)) {
			FillDiag(program.shader_hash, 2);
			g_pending_native_fill = std::make_tuple(descriptor.Base48(), size, packed_clear);
			return false;
		}
		FillDiag(program.shader_hash, 3);
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	if (image_clear) FillDiag(program.shader_hash, 1);
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.program->shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.program->bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	// An indirect dispatch's workgroup count is in GPU memory, and reading it back drained the
	// whole GPU for each such dispatch (NHL 27: ~19 K drains of ~3.5 ms per few minutes). The
	// shader indexes its LDS slice by workgroup, so bind room for as many workgroups as the
	// buffer holds; larger counts exceeded the buffer before as well.
	// KYTY_LDS_READBACK=1 (experiment switch, uncommitted) restores the readback.
	static const bool lds_readback = std::getenv("KYTY_LDS_READBACK") != nullptr;
	if (indirect_args != 0 && !lds_readback) {
		const uint64_t per_group = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(per_group == 0 || per_group > limit);
		bindings.shared_memory = {storage.Handle(), 0, limit / per_group * per_group};
		return;
	}
	if (indirect_args != 0) {
		const SyncReasonScope reason {"lds-indirect-args"};
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode, bool async_compute) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0] = thread_group_x;
	input_info.workgroup_counts[1] = thread_group_y;
	input_info.workgroup_counts[2] = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	const auto compute_program =
	    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	if (!compute_program) return;
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	g_pending_native_fill.reset();
	// Diagnostic (uncommitted, [gdsdiag]): NHL 27's strand-hair fine rasterizer
	// (cs ea94f09b2981eb43, 144x1x1; a GDS work queue: limit GDS[0], counter GDS[4]) hung the GPU
	// in run 61. Before the dispatch: the last CP writes to GDS[0..7]; after it completes: GDS[0..7].
	if (program.shader_hash == 0xea94f09b2981eb43ull) {
		const auto n = ++g_gds_hair_dispatches;
		{
			auto&       record = g_gds_hair_ring[n % 64];
			const auto& w      = g_gds_writes[0];
			record = {n, m_context.GetGpu().GetFrameNum(), w.kind, w.source, w.serial, 0, 0, false};
			auto& cache = m_context.GetBufferCache();
			m_context.GetCommandScheduler().DeferOperation([&cache, n] {
				auto& record = g_gds_hair_ring[n % 64];
				if (record.n != n) {
					return;
				}
				uint32_t gds[8] = {};
				cache.ReadGdsDwords(gds, 8);
				record.limit   = gds[0];
				record.counter = gds[4];
				record.done    = true;
			});
		}
		if (n <= 24 || n % 512 == 0) {
			char text[512] = {};
			int  used      = 0;
			for (uint32_t i = 0; i < 4 && used >= 0 && used < static_cast<int>(sizeof(text)); i++) {
				const auto& w = g_gds_writes[i];
				used += std::snprintf(text + used, sizeof(text) - static_cast<size_t>(used),
				                      " [%u]k%d:%08x@%llx#%llu", i * 4, w.kind, w.value,
				                      static_cast<unsigned long long>(w.source),
				                      static_cast<unsigned long long>(w.serial));
			}
			std::fprintf(stderr, "[gdsdiag] hair #%llu groups %ux%ux%u before:%s\n",
			             static_cast<unsigned long long>(n), thread_group_x, thread_group_y,
			             thread_group_z, text);
			std::fflush(stderr);
			auto& cache = m_context.GetBufferCache();
			m_context.GetCommandScheduler().DeferOperation([&cache, n] {
				uint32_t gds[8] = {};
				cache.ReadGdsDwords(gds, 8);
				std::fprintf(stderr,
				             "[gdsdiag] hair #%llu after: %08x %08x %08x %08x %08x %08x %08x %08x\n",
				             static_cast<unsigned long long>(n), gds[0], gds[1], gds[2], gds[3],
				             gds[4], gds[5], gds[6], gds[7]);
				std::fflush(stderr);
			});
		}
	}
	if (resources.uniform_fill.kind == ShaderRecompiler::IR::UniformFillKind::Buffer &&
	    !resources.specialization_reads.empty()) {
		FillDiag(program.shader_hash, 6);
	}
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
	                           thread_group_z, mode))) {
		ResetBindings();
		return;
	}

	if (use_thread_dimensions) {
		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = input_info.workgroup_counts[0];
		thread_group_y       = input_info.workgroup_counts[1];
		thread_group_z       = input_info.workgroup_counts[2];

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	const SyncShaderScope sync_shader {input_info.stage.program->shader_hash};
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);

	auto              vk_buffer        = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	if (has_storage_writes) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (GpuCrashDiag::Enabled()) {
		GpuCrashDiag::Record record {};
		record.op        = static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect);
		record.submit_id = submit_id;
		record.frame     = m_context.GetGpu().GetFrameNum();
		record.tick      = m_context.GetCommandScheduler().CurrentTick();
		record.hash[0]   = program.shader_hash;
		record.args[0]   = thread_group_x;
		record.args[1]   = thread_group_y;
		record.args[2]   = thread_group_z;
		record.args[3]   = mode;
		record.addr      = sh_ctx.GetCs().cs_regs.data_addr;
		GpuCrashDiag::Mark(vk_buffer, record);
	}
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	if (g_pending_native_fill) {
		const auto [base, size, value] = *g_pending_native_fill;
		g_gpu_fills[base]              = {size, value};
		g_pending_native_fill.reset();
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);

	// The removed host fence also ordered read-only dispatches before later writers.
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

void RenderExecutor::DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer,
                                      uint64_t args_addr, uint32_t mode, bool async_compute) {
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	// Experiment (uncommitted): KYTY_INDIRECT_AS_DIRECT=1 runs an indirect dispatch whose arguments
	// the CPU can read (not GPU-written) as a direct dispatch with those counts, so proven fills are
	// consumed or recorded like direct ones. NHL 27 fills color metadata this way (cs bed9da36...);
	// without a fill record, color-clear discovery drained the GPU (~20 ms) several times a second.
	static const bool as_direct = [] {
		const char* value = std::getenv("KYTY_INDIRECT_AS_DIRECT");
		return value != nullptr && value[0] == '1';
	}();
	if (as_direct) {
		vk::DispatchIndirectCommand known {};
		bool                        readable = false;
		{
			Common::LockGuard lock(m_context.GetMutex());
			readable = !m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(known)) &&
			           LibKernel::Memory::TryReadBacking(args_addr, &known, sizeof(known));
		}
		static std::atomic<uint64_t> total {0}, converted {0};
		const auto                   n = total.fetch_add(1, std::memory_order_relaxed) + 1;
		if (readable) converted.fetch_add(1, std::memory_order_relaxed);
		if (n % 4096 == 0) {
			std::fprintf(stderr, "[indirect2direct] %llu indirect dispatches, %llu run as direct\n",
			             static_cast<unsigned long long>(n),
			             static_cast<unsigned long long>(converted.load(std::memory_order_relaxed)));
			std::fflush(stderr);
		}
		if (readable) {
			DispatchDirect(submit_id, buffer, known.x, known.y, known.z, mode, async_compute);
			return;
		}
	}
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect), submit_id,
	                    static_cast<uint32_t>(args_addr), static_cast<uint32_t>(args_addr >> 32u),
	                    0, mode, buffer.GetShaders().GetCs().cs_regs.data_addr);
	Common::LockGuard lock(m_context.GetMutex());
	vk::DispatchIndirectCommand args {};
	if (!m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, sizeof(args)) &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	const auto compute_program = m_context.GetPipelineCache().GetComputeProgram(
	    cs_regs, buffer.GetRegisters().GetShaderRegisters(), input_info);
	if (!compute_program) return;
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	const SyncShaderScope sync_shader {input_info.stage.program->shader_hash};
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = *input_info.stage.program;
	if (input_info.stage.resources->uniform_fill.kind ==
	    ShaderRecompiler::IR::UniformFillKind::Buffer) {
		FillDiag(program.shader_hash, 7);
	}
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
	// Experiment (uncommitted): KYTY_DISPATCH_GUARD=0 disables the guard pass that zeroes indirect
	// dispatches whose group counts exceed the device limits or KYTY_DISPATCH_GUARD_MAX_GROUPS
	// (default 4M). Recorded before this dispatch's own bindings, which CommitBindings re-pushes.
	static const bool dispatch_guard = [] {
		const char* value = std::getenv("KYTY_DISPATCH_GUARD");
		return value == nullptr || value[0] != '0';
	}();
	vk::Buffer dispatch_buffer = args_buffer->Handle();
	uint64_t   dispatch_offset = args_offset;
	if (dispatch_guard) {
		if (m_dispatch_guard == nullptr) {
			const char*    text      = std::getenv("KYTY_DISPATCH_GUARD_MAX_GROUPS");
			const uint32_t max_total = text != nullptr && std::strtoul(text, nullptr, 10) != 0
			                               ? static_cast<uint32_t>(std::strtoul(text, nullptr, 10))
			                               : 4u * 1024u * 1024u;
			m_dispatch_guard = std::make_unique<DispatchArgsGuard>(
			    m_context.GetGraphics(), m_context.GetCommandScheduler(), max_total);
		}
		const auto slot = m_dispatch_guard->Record(buffer.Handle(), *args_buffer, args_offset);
		dispatch_buffer = slot.buffer;
		dispatch_offset = slot.offset;
	}
	RebindBuffers(bindings);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	const auto vk_buffer = buffer.Handle();
	const bool has_storage_writes = bindings.shared_memory.buffer != nullptr ||
	    HasShaderBufferWrites(input_info.stage) ||
	    std::any_of(program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.written && image.resource_class ==
		                                ShaderRecompiler::IR::ImageResourceClass::Storage;
	    });
	if (has_storage_writes) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {},
	                          1, &barrier, 0, nullptr, 0, nullptr);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	if (GpuCrashDiag::Enabled()) {
		GpuCrashDiag::Record record {};
		record.op        = static_cast<uint32_t>(CommandBufferDebugOp::DispatchIndirect);
		record.submit_id = submit_id;
		record.frame     = m_context.GetGpu().GetFrameNum();
		record.tick      = m_context.GetCommandScheduler().CurrentTick();
		record.hash[0]   = program.shader_hash;
		record.args[3]   = mode;
		record.addr      = cs_regs.cs_regs.data_addr;
		GpuCrashDiag::Mark(vk_buffer, record);
	}
	vk_buffer.dispatchIndirect(dispatch_buffer, dispatch_offset);
	{
		// Diagnostic (uncommitted, [bigdispatch]): run 70 stalled the GPU >10 s in an indirect
		// dispatch (cs d45140bb8cfa8799); pengui07's fork zeroes indirect dispatches over 4M groups
		// for the same reason. Copy the GPU-computed group counts into a host ring after the
		// dispatch and, once it completed, log counts above 100k groups with the shader.
		static std::unique_ptr<Buffer> args_log;
		static uint32_t                args_slot = 0;
		if (!args_log) {
			args_log = std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
			                                    MemoryUsage::Download, 0,
			                                    vk::BufferUsageFlagBits::eTransferDst, 4096u * 16u);
		}
		const uint32_t slot = (args_slot++ % 4096u) * 16u;
		args_log->CopyFrom(m_context.GetCommandScheduler().Current(), *args_buffer, args_offset, slot,
		                   sizeof(vk::DispatchIndirectCommand));
		auto* const    log  = args_log.get();
		const uint64_t hash = program.shader_hash;
		// The last GPU write marking that covered the args, at record time: its shader and range.
		uint64_t writer = 0, write_start = 0, write_size = 0;
		if (auto w = g_gpu_write_ranges.upper_bound(args_addr); w != g_gpu_write_ranges.begin()) {
			--w;
			if (w->first <= args_addr && args_addr < w->first + w->second.first) {
				writer      = w->second.second;
				write_start = w->first;
				write_size  = w->second.first;
			}
		}
		const bool gpu_written = m_context.GetBufferCache().HasGpuDirtyBytes(args_addr, 12);
		m_context.GetCommandScheduler().DeferOperation(
		    [log, slot, hash, args_addr, writer, write_start, write_size, gpu_written,
		     guard = m_dispatch_guard.get()] {
			if (!log->IsCoherent()) {
				log->Invalidate(0, log->Size());
			}
			uint32_t groups[3] = {};
			std::memcpy(groups, log->Mapped().data() + slot, sizeof(groups));
			const uint64_t total = uint64_t {groups[0]} * groups[1] * groups[2];
			static uint64_t big = 0, checked = 0, largest = 0;
			checked++;
			largest = std::max(largest, total);
			if (total > 100000u && big++ < 64) {
				std::fprintf(stderr,
				             "[bigdispatch] cs %016llx groups %u x %u x %u = %llu | args 0x%llx %s, "
				             "last GPU write by cs %016llx over 0x%llx+0x%llx\n",
				             static_cast<unsigned long long>(hash), groups[0], groups[1], groups[2],
				             static_cast<unsigned long long>(total),
				             static_cast<unsigned long long>(args_addr),
				             gpu_written ? "GPU-written" : "NOT GPU-written at record time",
				             static_cast<unsigned long long>(writer),
				             static_cast<unsigned long long>(write_start),
				             static_cast<unsigned long long>(write_size));
				std::fflush(stderr);
			}
			if (checked % 16384 == 0) {
				std::fprintf(stderr,
				             "[bigdispatch] %llu indirect dispatches checked, %llu above 100k groups, "
				             "largest %llu, zeroed by the guard %u\n",
				             static_cast<unsigned long long>(checked),
				             static_cast<unsigned long long>(big),
				             static_cast<unsigned long long>(largest),
				             guard != nullptr ? guard->Zeroed() : 0u);
				std::fflush(stderr);
			}
		});
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
