#include "graphics/guest_gpu/stageStats.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/experiments.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/runtimeFlags.h"
#include "common/threads.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/frameBreakdown.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/resourceMemo.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecord.h"
#include "graphics/host_gpu/renderer/pipeline/programStore.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// The driver validates the data by pipelineCacheUUID and looks pipelines up by content, so
	// data from another emulator build is safe to reuse.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	if (FrameBreakdown::g_spec_collect != nullptr) FrameBreakdown::SpecCollectRead(address, values.size_bytes());
	// Experiment GpuConditionsBothPaths: a branch condition on GPU-written bytes is left undecided.
	if (ShaderRecompiler::IR::ConditionReadsNonBlocking()) {
		return !values.empty() && Libs::LibKernel::Memory::TryReadCleanBufferBacking(
		                              address, values.data(), values.size_bytes());
	}
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

bool ReadShaderOrdinaryBacking(void*, uint64_t address, std::span<uint32_t> values) {
	// Ordinary scalar loads previously dereferenced the protected guest mapping.
	// Read only the requested bytes through its alias, synchronizing actual GPU
	// writes rather than unrelated bytes sharing the protection page. Non-backed
	// pointers retain the original memcpy/fault behavior; strict reads still fail.
	if (FrameBreakdown::g_spec_collect != nullptr) FrameBreakdown::SpecCollectRead(address, values.size_bytes());
	if (!values.empty() && Libs::LibKernel::Memory::TryReadBufferBacking(
	        address, values.data(), values.size_bytes())) {
		return true;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

bool PeekShaderCleanSpan(void*, uint64_t address, std::span<uint32_t> values) {
	// This capability declines dirty or unavailable ranges without synchronizing
	// them. The walker then performs its original ordered scalar reads.
	if (FrameBreakdown::g_spec_collect != nullptr) FrameBreakdown::SpecCollectRead(address, values.size_bytes());
	return !values.empty() && Libs::LibKernel::Memory::TryReadCleanBufferBacking(
	    address, values.data(), values.size_bytes());
}

bool ReadShaderNativeSpan(void*, uint64_t address, std::span<uint32_t> values) {
	if (!GuestGpu::IsGpuThread() || values.empty() ||
	    !GuestRange{address, values.size_bytes()}.Valid()) return false;
	// This completes the original ordinary native reads; it does not certify the
	// range clean. Protected GPU-owned bytes follow existing fault/readback handling.
	if (FrameBreakdown::g_spec_collect != nullptr) FrameBreakdown::SpecCollectRead(address, values.size_bytes());
	// After access begins, failures keep native fault behavior rather than returning false.
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

template <bool strict, bool native = false>
bool CaptureShaderSpan(void* data, uint64_t address, std::span<uint32_t> values) {
	const auto reader = native ? ReadShaderNativeSpan : PeekShaderCleanSpan;
	if (!reader(nullptr, address, values)) return false;
	auto& record = *static_cast<PipelineRecord::Program*>(data);
	auto& reads = strict ? record.reads : record.raw_reads;
	for (size_t index = 0; index < values.size(); ++index) {
		reads.push_back({address + index * sizeof(uint32_t), values[index], 1u});
	}
	return true;
}


bool ReplayRecordedWords(void* data, uint64_t address, std::span<uint32_t> values, bool strict) {
    const auto& record = *static_cast<const PipelineRecord::Program*>(data);
    return PipelineRecord::ReplayWords(record, address, values, strict);
}
bool ReplayRawWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return ReplayRecordedWords(data, address, values, false);
}
bool ReplayCleanWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return ReplayRecordedWords(data, address, values, true);
}
bool CaptureRecordedWords(void* data, uint64_t address, std::span<uint32_t> values, bool strict,
                          bool ordinary_backing = false) {
    auto& record = *static_cast<PipelineRecord::Program*>(data);
    const bool valid = strict ? ReadShaderGuestMemory(nullptr, address, values) : true;
    if (!strict) {
        if (ordinary_backing) ReadShaderOrdinaryBacking(nullptr, address, values);
        else std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
    }
    auto& reads = strict ? record.reads : record.raw_reads;
    for (size_t index = 0; index < values.size(); ++index)
        reads.push_back({address + index * 4, valid ? values[index] : 0u, valid ? 1u : 0u});
    return valid;
}
bool CaptureRawWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, false);
}
bool CaptureOrdinaryBackingWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, false, true);
}
bool CaptureCleanWords(void* data, uint64_t address, std::span<uint32_t> values) {
    return CaptureRecordedWords(data, address, values, true);
}

// NHL26 debugging: KYTY_DBG_DUMP_SHADERS=<hex>[,<hex>...] dumps just those shaders.
static bool ForceDumpShader(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		if (const char* v = std::getenv("KYTY_DBG_DUMP_SHADERS"); v != nullptr) {
			for (const char* p = v; *p != '\0';) {
				char*      end   = nullptr;
				const auto value = std::strtoull(p, &end, 16);
				if (end == p) break;
				list.push_back(value);
				p = (*end == ',') ? end + 1 : end;
			}
		}
		return list;
	}();
	return std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
}
bool ReadAheadPlain(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && static_cast<GuestReadLog*>(userdata)->Read(
	                              address, values.data(), values.size_bytes(), false);
}

bool ReadAheadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && static_cast<GuestReadLog*>(userdata)->Read(address, values.data(),
	                                                                     values.size_bytes(), true);
}

bool ReadAheadShaderState(void* userdata, uint64_t address, void* data, uint64_t size) {
	return static_cast<GuestReadLog*>(userdata)->Read(address, data, size, false);
}

} // namespace

bool GuestReadLog::Read(uint64_t address, void* data, uint64_t size, bool strict) {
	if (size == 0 || size > UINT32_MAX ||
	    !Libs::LibKernel::Memory::TryReadBacking(address, data, size)) {
		m_failed = true;
		return false;
	}
	const auto offset = static_cast<uint32_t>(m_bytes.size());
	m_bytes.insert(m_bytes.end(), static_cast<const uint8_t*>(data),
	               static_cast<const uint8_t*>(data) + size);
	// Contiguous reads of the same kind are checked as one (strict reads up to the size the
	// check reads at once).
	if (!m_entries.empty()) {
		auto& last = m_entries.back();
		if (last.strict == strict && last.address + last.size == address &&
		    last.offset + last.size == offset &&
		    (!strict || last.size + size <= StrictCheckBytes)) {
			last.size += static_cast<uint32_t>(size);
			return true;
		}
	}
	m_entries.push_back({address, static_cast<uint32_t>(size), offset, strict});
	return true;
}

// Most logged reads are a few dwords: compare those without a call.
static bool SameGuestBytes(const void* guest, const uint8_t* recorded, uint32_t size) {
	const auto load = [](const void* address, uint32_t offset, auto value) {
		std::memcpy(&value, static_cast<const uint8_t*>(address) + offset, sizeof(value));
		return value;
	};
	switch (size) {
		case 4: return load(guest, 0, uint32_t {}) == load(recorded, 0, uint32_t {});
		case 8: return load(guest, 0, uint64_t {}) == load(recorded, 0, uint64_t {});
		case 16:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {}))) == 0;
		case 32:
			return ((load(guest, 0, uint64_t {}) ^ load(recorded, 0, uint64_t {})) |
			        (load(guest, 8, uint64_t {}) ^ load(recorded, 8, uint64_t {})) |
			        (load(guest, 16, uint64_t {}) ^ load(recorded, 16, uint64_t {})) |
			        (load(guest, 24, uint64_t {}) ^ load(recorded, 24, uint64_t {}))) == 0;
		default: return std::memcmp(guest, recorded, size) == 0;
	}
}

bool GuestReadLog::StillValid(bool without_faults) const {
	if (m_failed) {
		return false;
	}
	std::array<uint8_t, StrictCheckBytes> buffer {};
	// Consecutive entries mostly lie on one page: its protection is looked up once.
	uint64_t   protected_page = ~uint64_t {0};
	bool       page_protected = false;
	const auto read_protected = [&](uint64_t address, uint64_t size) {
		const auto page = address & ~uint64_t {0xfff};
		if (page != ((address + size - 1) & ~uint64_t {0xfff})) {
			return Libs::LibKernel::Memory::IsGpuReadProtected(address, size);
		}
		if (page != protected_page) {
			protected_page = page;
			page_protected = Libs::LibKernel::Memory::IsGpuReadProtected(page, 0x1000);
		}
		return page_protected;
	};
	// The GPU-dirty checks of the entries too: once per page (two tree lookups per entry were a
	// third of this check; execution time per dispatch -14% in a paused A/B).
	std::array<uint64_t, 2> clean_page {~uint64_t {0}, ~uint64_t {0}};
	std::array<bool, 2>     page_clean {};
	const auto              known_clean = [&](uint64_t address, uint64_t size, bool gpu_images) {
		const auto page = address & ~uint64_t {0xfff};
		if (page != ((address + size - 1) & ~uint64_t {0xfff})) {
			return false;
		}
		const auto kind = gpu_images ? 1 : 0;
		if (page != clean_page[kind]) {
			clean_page[kind] = page;
			page_clean[kind] = Libs::LibKernel::Memory::IsGpuCleanRange(page, 0x1000, gpu_images);
		}
		return page_clean[kind];
	};
	for (const auto& entry: m_entries) {
		const auto* recorded = m_bytes.data() + entry.offset;
		if (entry.strict) {
			// As specialization memory is read on the execution thread: GPU-owned bytes fail.
			if (entry.size > buffer.size() ||
			    !(known_clean(entry.address, entry.size, true)
			          ? Libs::LibKernel::Memory::TryReadBacking(entry.address, buffer.data(), entry.size)
			          : Libs::LibKernel::Memory::TryReadGpuCleanBacking(entry.address, buffer.data(),
			                                                            entry.size)) ||
			    std::memcmp(buffer.data(), recorded, entry.size) != 0) {
				return false;
			}
		} else {
			// Experiment FaultPageDirect: on a page that faulted before (the GPU wrote other
			// bytes of it), clean bytes are compared in the backing instead of faulting again.
			// Dispatch read-ahead checks never fault: their pages change every frame (ring
			// buffers), so the first read of each faulted (about 5 ms each, 15% of the thread).
			// Draw checks look at the page protection too: their plain loads of such pages took an
			// exception each (2.7% of the execution thread; the 16-page fault cache thrashes).
			if (Common::Experiments::Enabled(Common::Experiments::FaultPageDirect) &&
			    (without_faults || Libs::LibKernel::Memory::IsReadFaultPage(entry.address, entry.size) ||
			     read_protected(entry.address, entry.size))) {
				const auto compare =
				    known_clean(entry.address, entry.size, false)
				        ? Libs::LibKernel::Memory::CompareCleanBacking(entry.address, recorded, entry.size)
				        : Libs::LibKernel::Memory::TryValidateCleanBacking(entry.address, recorded, entry.size);
				if (compare == Libs::LibKernel::Memory::CleanBackingCompare::Equal) {
					continue;
				}
				if (compare == Libs::LibKernel::Memory::CleanBackingCompare::Different ||
				    (without_faults && compare == Libs::LibKernel::Memory::CleanBackingCompare::Dirty)) {
					return false;
				}
			}
			// A plain load, as the execution thread's own resolution reads: GPU-owned pages
			// fault and read the GPU's bytes back.
			if (!SameGuestBytes(reinterpret_cast<const void*>(entry.address), recorded, entry.size)) {
				return false;
			}
		}
	}
	return true;
}

namespace {

class ProgramLockGuard {
public:
	explicit ProgramLockGuard(std::atomic_flag& flag): m_flag(flag) {
		if (!m_flag.test_and_set(std::memory_order_acquire)) {
			return;
		}
		const bool stats = StageStats::Enabled();
		const auto start = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		do {
			while (m_flag.test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		} while (m_flag.test_and_set(std::memory_order_acquire));
		if (stats && StageStats::ProgramLockCounter() != StageStats::CounterCount) {
			StageStats::Add(StageStats::ProgramLockCounter(), start);
		}
	}
	~ProgramLockGuard() { m_flag.clear(std::memory_order_release); }
	ProgramLockGuard(const ProgramLockGuard&)            = delete;
	ProgramLockGuard& operator=(const ProgramLockGuard&) = delete;

private:
	std::atomic_flag& m_flag;
};

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled() && !ForceDumpShader(shader_hash)) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled() && !ForceDumpShader(shader_hash)) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::deque<Permutation>                    permutations;
		ResourceMemo                               resource_memo;
		ShaderRecompiler::IR::ResourceSpecializationCache specialization_cache;
		ShaderRecompiler::IR::ResourceDelta               delta; // KYTY_SRT_COMPILED: this program's previous outputs
		size_t                                     last_permutation = std::numeric_limits<size_t>::max();
		bool                                       native_plan_checked = false;
		bool                                       native_plan_allowed = false;
		// Experiments AheadUnlocked / ComputeUnlocked: a thread materializes this entry's plan
		// without the program lock (the plan's scratch is the entry's own); set and cleared
		// under the lock. The resolve thread skips a busy entry, the execution thread waits.
		std::atomic<bool>                          busy {false};
		uint64_t                                   refresh_ns = 0;
		uint64_t                                   refresh_calls = 0;
		uint64_t                                   refresh_hits = 0;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		// KYTY_PERF_STATS: shader permutation compile time (SPIR-V generation and module creation).
		FrameBreakdown::Timer compile_timer(FrameBreakdown::g_counters.compile_ns);
		FrameBreakdown::Add(FrameBreakdown::g_counters.compile_count);
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
		// KYTY_DBG_DUMP_SHADERS: the guest disassembly and the IR next to the SPIR-V.
		if (ForceDumpShader(options.shader_hash) && !result.decoded_dump.empty()) {
			static std::atomic_int text_id = 0;
			const auto path = Config::GetShaderLogFolder() /
			                  fmt::format("{:04d}_ir_{}_{:016x}.txt", text_id++, stage_name, options.shader_hash);
			Common::File::CreateDirectories(path.parent_path());
			Common::File file(path);
			if (!file.IsInvalid()) {
				file.Write(result.decoded_dump.data(), result.decoded_dump.size());
				file.Write(result.ir_dump.data(), result.ir_dump.size());
			}
		}

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = next_shader_id.fetch_add(1) + 1, .module = module},
		};
	}

	template <typename InputInfo>
	std::pair<ShaderRecompiler::CompileOptions, const char*>
	MakeCompileOptions(ShaderType stage, const ShaderParams& params, InputInfo& input_info) {
		const auto           user_data = std::span(params.user_data).first(params.user_data_count);
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label      = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex:
				label      = "ShaderRecompiler VS";
				stage_name = "vs";
				break;
			case ShaderType::Mesh:
				label      = "ShaderRecompiler MS";
				stage_name = "ms";
				break;
			case ShaderType::Local:
				label      = "ShaderRecompiler LS";
				stage_name = "ls";
				break;
			case ShaderType::TessellationControl:
				label      = "ShaderRecompiler HS";
				stage_name = "hs";
				break;
			case ShaderType::TessellationEvaluation:
				label      = "ShaderRecompiler DS";
				stage_name = "ds";
				break;
			case ShaderType::Pixel:
				label      = "ShaderRecompiler PS";
				stage_name = "ps";
				break;
			case ShaderType::Compute:
				label      = "ShaderRecompiler CS";
				stage_name = "cs";
				break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code   = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent ||
		                      ForceDumpShader(params.hash);
		options.early_dump  = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		const char* scalar = std::getenv("KYTY_GPU_SCALAR_DATA");
		options.gpu_scalar_data = scalar != nullptr && scalar[0] == '1' && scalar[1] == 0;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size      = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return {options, stage_name};
	}

	// Resolves a stage ahead of execution: only programs and permutations that exist, into the
	// storage of the resolution, reading guest memory through its log.
	template <typename InputInfo>
	ShaderProgram GetAhead(const ShaderParams& params, InputInfo& input_info,
	                       uint32_t& push_data_cursor, ProgramResolution& ahead) {
		ShaderType stage;
		constexpr bool compute = std::is_same_v<InputInfo, ShaderComputeInputInfo>;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (compute) {
			stage = ShaderType::Compute;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderPixelInputInfo>);
			stage = ShaderType::Pixel;
		}
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		const auto entry = programs.find(lookup_key);
		// Compute programs are compiled by the execution thread itself.
		if constexpr (!compute) {
			if (entry == programs.end() && ahead.stages != ProgramResolution::MaxStages) {
				QueueCompile(stage, params, input_info, push_data_cursor, nullptr);
			}
		}
		// SWAPPC programs link the functions they call from guest memory: the execution thread
		// resolves them.
		if (entry == programs.end() || ahead.stages == ProgramResolution::MaxStages ||
		    entry->second.call_source || entry->second.busy.load(std::memory_order_acquire)) {
			ahead.reads.Fail();
			return {};
		}
		const auto                             stage_index    = ahead.stages++;
		auto&                                  resources      = ahead.resources[stage_index];
		auto&                                  specialization = ahead.specializations[stage_index];
		ShaderRecompiler::IR::SrtRuntime runtime {
		    .user_data                  = std::span(params.user_data).first(params.user_data_count),
		    .shader_base                = params.Base(),
		    .read_memory                = ReadAheadPlain,
		    .userdata                   = &ahead.reads,
		    .read_specialization_memory = ReadAheadStrict,
		};
		if constexpr (compute) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		// Contiguous resource-table words are read with one call, as the execution thread reads them
		// (the resolve thread read each dword with its own call chain, 25% of the thread; and each
		// switch between the two rebuilt the plan's read runs).
		runtime.batch_reads = env_batch_reads || Common::Experiments::Enabled(Common::Experiments::SrtBatchReads);
		auto& source = entry->second;
		bool  materialized = false;
		// Experiment AheadUnlocked: the execution thread may take the lock meanwhile; it keeps
		// off this entry until the materialization is done.
		if (program_lock != nullptr && Common::Experiments::Enabled(Common::Experiments::AheadUnlocked)) {
			source.busy.store(true, std::memory_order_release);
			UnlockPrograms();
			materialized =
			    ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime, resources, specialization);
			LockPrograms();
			source.busy.store(false, std::memory_order_release);
			// The other thread may have used the shared lookup key meanwhile.
			lookup_key.stage           = stage;
			lookup_key.hash            = params.hash;
			lookup_key.user_data_count = params.user_data_count;
			lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
			BuildStageStaticKey(input_info, lookup_key.static_state);
		} else {
			materialized =
			    ShaderRecompiler::IR::MaterializeResources(source.resource_plan, runtime, resources, specialization);
		}
		if (!materialized) {
			ahead.reads.Fail();
			return {};
		}
		const auto permutation =
		    std::ranges::find_if(source.permutations, [&](const Permutation& candidate) {
			    const auto& layout = candidate.program.bindings;
			    return layout.push_data_start_dword ==
			               ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
			                                                        layout.ShaderDataDwords()) &&
			           candidate.specialization == specialization;
		    });
		if (permutation == source.permutations.end()) {
			if constexpr (!compute) {
				QueueCompile(stage, params, input_info, push_data_cursor, &specialization);
			}
			ahead.reads.Fail();
			return {};
		}
		input_info.stage = {.program = &permutation->program, .resources = &resources};
		permutation->program.bindings.AdvancePushData(push_data_cursor);
		return permutation->handle;
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		// NHL debugging: shaders whose CFG build failed under KYTY_DBG_SKIP_BAD_SHADERS=1.
		if (!failed_shaders.empty()) {
			if (const auto failed = failed_shaders.find(params.hash); failed != failed_shaders.end()) {
				if (failed->second++ < 4u) {
					std::printf("NHL26DBG: skipping use of shader 0x%016" PRIx64
					            " (CFG build failed)\n", params.hash);
					std::fflush(stdout);
				}
				return {};
			}
		}

		ApplyExperiments();
		BufferCache::NoteReadbackShader(params.hash);
		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		// Diagnostics (KYTY_SPECULATE_DIRTY_READS stats): which shader the following guest reads serve.
		Libs::LibKernel::Memory::SetReadContext(params.hash);
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		static const bool reuse_enabled = [] {
			const char* value = std::getenv("KYTY_PROGRAM_RESOURCE_CACHE");
			return value != nullptr && value[0] == '1' && value[1] == 0;
		}();
		++lookups;
		auto& last = last_sources[static_cast<size_t>(stage)];
		SourceEntry* entry = nullptr;
		if (reuse_enabled && last.key != nullptr && *last.key == lookup_key) {
			entry = last.source;
			++source_hits;
		} else if (auto found = programs.find(lookup_key); found != programs.end()) {
			entry = &found->second;
			last = {&found->first, entry};
		}
		// Experiments AheadUnlocked / ComputeUnlocked: another thread materializes this entry
		// without the lock; wait without it, then look up again.
		if (entry != nullptr && entry->busy.load(std::memory_order_acquire)) {
			UnlockPrograms();
			while (entry->busy.load(std::memory_order_acquire)) {
				Common::SpinPause();
			}
			LockPrograms();
			return Get(params, input_info, push_data_cursor);
		}
		const auto runtime_flags = Common::RuntimeFlags::Current();
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_memory = runtime_flags.shader_clean_reads
		                       ? ReadShaderOrdinaryBacking : nullptr,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .try_read_memory_span = runtime_flags.srt_scalar_batch
		                                ? (runtime_flags.shader_clean_reads
		                                       ? PeekShaderCleanSpan : ReadShaderNativeSpan) : nullptr,
		    .try_read_specialization_span = runtime_flags.srt_scalar_batch
		                                        ? PeekShaderCleanSpan : nullptr,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
		}
		if (replaying != nullptr) {
            runtime.read_memory = ReplayRawWords;
            runtime.read_specialization_memory = ReplayCleanWords;
            runtime.userdata = const_cast<PipelineRecord::Program*>(replaying);
            // Frozen PipelineRecords replay through their domain readers. Never
            // retain a live-memory capability after replacing its userdata.
            runtime.try_read_memory_span = nullptr;
            runtime.try_read_specialization_span = nullptr;
        }
        const bool memo_enabled = reuse_enabled && replaying == nullptr && !runtime_flags.resource_memo_bypass;
		runtime.batch_reads = batch_reads;
        const auto refresh = [&](SourceEntry& source, bool& reused) {
            const auto begin = resource_timing ? std::chrono::steady_clock::now()
                                               : std::chrono::steady_clock::time_point {};
            auto observed_runtime = runtime;
            observed_runtime.specialization_cache = runtime_flags.resource_spec_cache
                ? &source.specialization_cache : nullptr;
            // Frozen pipeline-record readers see single-dword requests only; never merge there.
            observed_runtime.allow_compiled = replaying == nullptr;
            observed_runtime.delta = runtime_flags.srt_compiled && replaying == nullptr ? &source.delta : nullptr;
            bool ok = source.resource_memo.Refresh(source.resource_plan, observed_runtime,
                source.resources, source.specialization, memo_enabled, reused);
            if (!ok && replaying == nullptr) {
                // Port diagnostic: name the shader, then retry on the pre-port reference walker
                // without the memo, delta or specialization cache. A failed Refresh already
                // dropped the memo's entries.
                auto reference                 = runtime;
                reference.allow_compiled       = false;
                reference.delta                = nullptr;
                reference.specialization_cache = nullptr;
                ok = ShaderRecompiler::IR::MaterializeResources(source.resource_plan, reference,
                    source.resources, source.specialization);
                reused = false;
                static std::atomic<uint32_t> logged {0};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
                    std::printf("NHL27GTAPORT: materialization failed shader=0x%016" PRIx64
                                " stage=%u user_data=%u reference_walker=%s\n",
                                params.hash, static_cast<unsigned>(stage), params.user_data_count,
                                ok ? "ok" : "failed");
                    std::fflush(stdout);
                }
            }
            if (resource_timing) {
                source.refresh_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - begin).count();
                ++source.refresh_calls;
                source.refresh_hits += reused;
            }
            if (!ok && replaying == nullptr) EXIT("Resource materialization failed\n");
            return ok;
        };
        const auto native_plan_safe = [&](const ShaderRecompiler::IR::ResourcePlan& plan) {
			if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
				if (input_info.native_draw.Enabled()) {
					for (const auto& value: plan.value_storage) {
						if (value.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::GetUserData) continue;
						const auto reg = ShaderRecompiler::IR::RegIndex(value.Arg(0).ScalarRegister());
						if (std::ranges::find(input_info.native_draw.registers, reg) != input_info.native_draw.registers.end()) return false;
					}
				}
			}
			return true;
		};
		const auto check_native_plan = [&](SourceEntry& source) {
			if (reuse_enabled && source.native_plan_checked) return source.native_plan_allowed;
			const bool allowed = native_plan_safe(source.resource_plan);
			if (reuse_enabled) {
				// Native SGPR mappings belong to ProgramKey; the plan graph is immutable.
				source.native_plan_checked = true;
				source.native_plan_allowed = allowed;
			}
			return allowed;
		};
		// A permutation matches the push-data slot, the specialization and, for shaders with a
		// SWAPPC call, the callee code it was linked against.
		const auto permutation_matches = [&](const SourceEntry& source, const Permutation& candidate) {
			const auto& layout = candidate.program.bindings;
			return layout.push_data_start_dword ==
			           ShaderRecompiler::IR::PushData::StartFor(push_data_cursor,
			                                                    layout.ShaderDataDwords()) &&
			       candidate.specialization == source.specialization &&
			       (!source.call_source ||
			        std::ranges::equal(candidate.function_code, source.FunctionCode()));
		};
		bool source_changed = false;
		if (entry != nullptr && entry->call_source) {
			// The callee lives in guest memory; a changed callee retranslates the program below.
			source_changed = entry->RefreshCallSource(runtime);
		}
		if (entry != nullptr && !source_changed) {
			if (!check_native_plan(*entry)) return {};
			bool reused    = false;
			bool refreshed = false;
			// Experiment ComputeUnlocked: the resolve thread may look up graphics programs while
			// the execution thread materializes a compute program (the entry is marked busy).
			if (std::is_same_v<InputInfo, ShaderComputeInputInfo> && program_lock != nullptr &&
			    replaying == nullptr && Common::Experiments::Enabled(Common::Experiments::ComputeUnlocked)) {
				entry->busy.store(true, std::memory_order_release);
				UnlockPrograms();
				refreshed = refresh(*entry, reused);
				LockPrograms();
				entry->busy.store(false, std::memory_order_release);
				lookup_key.stage           = stage;
				lookup_key.hash            = params.hash;
				lookup_key.user_data_count = params.user_data_count;
				lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
				BuildStageStaticKey(input_info, lookup_key.static_state);
			} else {
				refreshed = refresh(*entry, reused);
			}
			if (!refreshed) return {};
			resource_hits += reused;
			materializations += !reused;
			if (memo_enabled && !reused) ++resource_misses[static_cast<size_t>(entry->resource_memo.LastMiss())];
			if (reuse_enabled && entry->last_permutation < entry->permutations.size()) {
				const auto& candidate = entry->permutations[entry->last_permutation];
				if (permutation_matches(*entry, candidate)) {
					++permutation_hits;
					input_info.stage = {.program = &candidate.program, .resources = &entry->resources};
					candidate.program.bindings.AdvancePushData(push_data_cursor);
					PrintStats(reuse_enabled);
					return candidate.handle;
				}
			}
			if (const auto permutation = std::ranges::find_if(
			        entry->permutations,
			        [&](const Permutation& candidate) { return permutation_matches(*entry, candidate); });
			    permutation != entry->permutations.end()) {
				entry->last_permutation = static_cast<size_t>(permutation - entry->permutations.begin());
				input_info.stage = {.program   = &permutation->program,
				                    .resources = &entry->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				PrintStats(reuse_enabled);
				return permutation->handle;
			}
		}

		// A worker may be compiling this program: use it once it is there.
		if (WaitForCompile()) {
			return Get(params, input_info, push_data_cursor);
		}
		const auto [options, stage_name] = MakeCompileOptions(stage, params, input_info);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		const auto push_data_start = push_data_cursor;
		ShaderRecompiler::ShaderSource source;
		if (entry == nullptr) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == nullptr ? &source : entry->call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr && current->call) {
			if (entry == nullptr) ShaderRecompiler::RefreshShaderSource(*current, runtime);
			translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
			translated.program.source_reads = current->reads;
		} else if (current != nullptr) {
			translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (translated.build_failed) {
			// NHL debugging: KYTY_DBG_SKIP_BAD_SHADERS=1 turned the CFG failure into a skipped draw.
			failed_shaders.emplace(params.hash, 0u);
			return {};
		}
		if (entry == nullptr) {
			auto retained = source.call
			                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
			                    : nullptr;
			const auto found = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			entry = &found->second;
			last = {&found->first, entry};
			if (!check_native_plan(*entry)) return {};
			bool reused = false;
			if (!refresh(*entry, reused)) return {};
			++materializations;
		} else if (source_changed) {
			// New callee: the plan's reads changed, so every plan-derived memo starts over.
			entry->resource_plan        = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			entry->resource_memo        = ResourceMemo {};
			entry->specialization_cache = {};
			entry->delta                = {};
			entry->native_plan_checked  = false;
			entry->last_permutation     = std::numeric_limits<size_t>::max();
			if (!check_native_plan(*entry)) return {};
			bool reused = false;
			if (!refresh(*entry, reused)) return {};
			++materializations;
			if (const auto permutation = std::ranges::find_if(
			        entry->permutations,
			        [&](const Permutation& candidate) { return permutation_matches(*entry, candidate); });
			    permutation != entry->permutations.end()) {
				entry->last_permutation = static_cast<size_t>(permutation - entry->permutations.begin());
				input_info.stage = {.program = &permutation->program, .resources = &entry->resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

        PipelineRecord::Program recorded;
        // Programs the warm-up record already holds are not appended again (record_keys).
        bool recording = record != nullptr && record->IsOpen() && replaying == nullptr &&
                         !record_keys.contains(lookup_key) &&
                         PipelineRecord::IsRecordableStage(stage) &&
                         !entry->call_source;
        if (recording) {
            recorded.stage = stage; recorded.hash = params.hash;
            recorded.shader_base = params.Base(); recorded.push_data_cursor = push_data_cursor;
            recorded.code.assign(params.code.begin(), params.code.end());
            recorded.back_code.assign(params.back_code.begin(), params.back_code.end());
            recorded.user_data.assign(user_data.begin(), user_data.end());
            if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) recorded.vertex = input_info;
            else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) recorded.pixel = input_info;
            else recorded.compute = input_info;
            auto capture_runtime = runtime;
            capture_runtime.userdata = &recorded;
            capture_runtime.read_memory = runtime.read_memory == ReadShaderOrdinaryBacking
                ? CaptureOrdinaryBackingWords : CaptureRawWords;
            capture_runtime.read_specialization_memory = CaptureCleanWords;
            capture_runtime.try_read_memory_span = runtime.try_read_memory_span != nullptr
                ? (runtime.try_read_memory_span == ReadShaderNativeSpan
                       ? CaptureShaderSpan<false, true> : CaptureShaderSpan<false>) : nullptr;
            capture_runtime.try_read_specialization_span = runtime.try_read_specialization_span != nullptr
                ? CaptureShaderSpan<true> : nullptr;
            ShaderRecompiler::IR::ResourceSnapshot captured;
            ShaderRecompiler::IR::ResourceSpecialization captured_specialization;
            recording = ShaderRecompiler::IR::MaterializeResources(entry->resource_plan, capture_runtime,
                                                                  captured, captured_specialization) &&
                        captured_specialization == entry->specialization;
        }
        entry->permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->specialization, push_data_cursor,
		    entry->FunctionCode()));
		entry->last_permutation = entry->permutations.size() - 1;
		const auto& permutation = entry->permutations.back();
        if (recording) ordinals[permutation.handle.id] = record->AppendProgram(recorded);
		Store(lookup_key, params, input_info, push_data_start, permutation.specialization);
		input_info.stage = {.program = &permutation.program, .resources = &entry->resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}
	void PrintStats(bool enabled) {
		static const bool stats = [] {
			const char* value = std::getenv("KYTY_PERF_STATS");
			return value != nullptr && value[0] == '1';
		}();
		if ((!stats && !resource_timing) || (lookups & 16383u) != 0) return;
		const auto now = std::chrono::steady_clock::now();
		if (now - last_stats < std::chrono::seconds(5)) return;
		if (stats) std::printf("NHL27PROGRAM: cache=%d lookups=%llu source_hits=%llu resource_hits=%llu materialize=%llu perm_hits=%llu\n",
		    enabled, static_cast<unsigned long long>(lookups), static_cast<unsigned long long>(source_hits),
		    static_cast<unsigned long long>(resource_hits), static_cast<unsigned long long>(materializations),
		    static_cast<unsigned long long>(permutation_hits));
		if (stats) {
			const auto spec = ShaderRecompiler::IR::GetResourceSpecializationCacheStats();
			std::printf("NHL27SPECCACHE: enabled=%d calls=%llu buffers=%llu/%llu images=%llu/%llu indirect_uncached=%llu\n",
			    Common::RuntimeFlags::Current().resource_spec_cache,
			    static_cast<unsigned long long>(spec.calls),
			    static_cast<unsigned long long>(spec.buffer_hits), static_cast<unsigned long long>(spec.buffer_misses),
			    static_cast<unsigned long long>(spec.image_hits), static_cast<unsigned long long>(spec.image_misses),
			    static_cast<unsigned long long>(spec.indirect_uncached));
		}
		if (stats && enabled) {
			std::printf("NHL27PROGRAMMISS: cold=%llu base=%llu regs=%llu groups=%llu read_failed=%llu memory_changed=%llu\n",
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Cold)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::ShaderBase)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Registers)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::Workgroups)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::ReadFailed)]),
			    static_cast<unsigned long long>(resource_misses[static_cast<size_t>(ResourceMemo::Miss::MemoryChanged)]));
			if (ResourceMissDiag::Enabled()) {
				auto& d = ResourceMissDiag::Get();
				const auto get = [](const std::atomic<uint64_t>& v) {
					return static_cast<unsigned long long>(v.load(std::memory_order_relaxed));
				};
				std::printf("NHL27MISSDIAG: misses=%llu changed=1:%llu,2:%llu,3-4:%llu,5+:%llu same: images=%llu samplers=%llu "
				            "buffers=%llu buffers_ignoring_base=%llu flat_srt=%llu spec=%llu fill=%llu "
				            "all_but_buffer_base=%llu identical=%llu first_reg:",
				    get(d.misses), get(d.changed[0]), get(d.changed[1]), get(d.changed[2]), get(d.changed[3]),
				    get(d.same_images), get(d.same_samplers), get(d.same_buffers), get(d.same_buffers_masked),
				    get(d.same_flat), get(d.same_spec), get(d.same_fill), get(d.all_but_buffer_base), get(d.identical));
				for (size_t i = 0; i < d.first_index.size(); ++i) {
					if (get(d.first_index[i]) != 0) std::printf(" %zu=%llu", i, get(d.first_index[i]));
				}
				std::printf("\n");
			}
			const auto reuse_trace = GetResourceReuseTraceStats();
			if (reuse_trace.checked != 0 || reuse_trace.evictions != 0) {
				std::printf("NHL27REUSETRACE: evictions=%llu checked=%llu register_recur=%llu distance16=%llu distance32=%llu distance64=%llu distance128=%llu\n",
				    static_cast<unsigned long long>(reuse_trace.evictions), static_cast<unsigned long long>(reuse_trace.checked),
				    static_cast<unsigned long long>(reuse_trace.matched), static_cast<unsigned long long>(reuse_trace.distance16),
				    static_cast<unsigned long long>(reuse_trace.distance32), static_cast<unsigned long long>(reuse_trace.distance64),
				    static_cast<unsigned long long>(reuse_trace.distance128));
			}
			const auto tape = ShaderRecompiler::IR::GetSrtTapeStats();
			const auto filter = GetResourceHashFilterStats();
			if (filter.eligible != 0) {
				std::printf("NHL27RESOURCEFILTER: eligible=%llu rejected=%llu hash_match=%llu\n",
				    static_cast<unsigned long long>(filter.eligible),
				    static_cast<unsigned long long>(filter.rejected),
				    static_cast<unsigned long long>(filter.matched));
			}
			std::printf("NHL27SRTTAPE: runs=%llu failed=%llu walker=%llu built=%llu unsupported=%llu ops=%llu cfg_roots=%llu cfg_fallback=%llu cfg_checked=%llu cfg_matched=%llu cfg_mismatch=%llu cfg_inconclusive=%llu trace_hits=%llu trace_misses=%llu trace_recorded=%llu\n",
			    static_cast<unsigned long long>(tape.runs),
			    static_cast<unsigned long long>(tape.failed),
			    static_cast<unsigned long long>(tape.walker),
			    static_cast<unsigned long long>(tape.built),
			    static_cast<unsigned long long>(tape.unsupported),
			    static_cast<unsigned long long>(tape.ops),
			    static_cast<unsigned long long>(tape.cfg_runs),
			    static_cast<unsigned long long>(tape.cfg_fallback),
			    static_cast<unsigned long long>(tape.cfg_checked),
			    static_cast<unsigned long long>(tape.cfg_matched),
			    static_cast<unsigned long long>(tape.cfg_mismatch),
			    static_cast<unsigned long long>(tape.cfg_inconclusive),
			    static_cast<unsigned long long>(tape.cfg_trace_hits),
			    static_cast<unsigned long long>(tape.cfg_trace_misses),
			    static_cast<unsigned long long>(tape.cfg_trace_recorded));
		}
		if (resource_timing) {
			std::vector<std::pair<const ProgramKey*, SourceEntry*>> measured;
			uint64_t total_ns = 0;
			for (auto& [key, source]: programs) {
				total_ns += source.refresh_ns;
				if (source.refresh_calls != 0) measured.emplace_back(&key, &source);
			}
			std::ranges::sort(measured, {}, [](const auto& item) {
				return UINT64_MAX - item.second->refresh_ns;
			});
			std::printf("NHL27RESOURCE: window=%.2f s refresh=%.2f ms sources=%zu\n",
			    std::chrono::duration<double>(now - last_stats).count(), total_ns / 1e6, measured.size());
			for (size_t i = 0; i < std::min<size_t>(15, measured.size()); ++i) {
				const auto& [key, source] = measured[i];
				const auto& plan = source->resource_plan;
				const auto batches = std::ranges::count_if(plan.srt_batch_counts, [](uint32_t count) { return count > 1; });
				std::printf("NHL27RESOURCE: hash=%016llx stage=%u ms=%.2f n=%llu hits=%llu avg=%.2f us values=%zu srt=%zu sources=%zu flat=%zu batch_starts=%zu\n",
				    static_cast<unsigned long long>(key->hash), static_cast<unsigned>(key->stage),
				    source->refresh_ns / 1e6, static_cast<unsigned long long>(source->refresh_calls),
				    static_cast<unsigned long long>(source->refresh_hits),
				    source->refresh_ns / 1e3 / source->refresh_calls, plan.value_storage.size(),
				    plan.srt_reads.size(), plan.descriptor_sources.size(), source->resources.flattened_srt.size(), static_cast<size_t>(batches));
			}
			for (const auto& item: measured) {
				item.second->refresh_ns = item.second->refresh_calls = item.second->refresh_hits = 0;
			}
			// Diagnostic A/B control runs on the sole cache-owning GPU thread. Changing
			// policy invalidates memo states and retains compiled shaders/pipelines.
			if (!cache_control_file.empty()) {
				unsigned capacity = 0;
				std::ifstream control(cache_control_file);
				if (control >> capacity && (capacity == 1 || capacity == 16)) {
					if (capacity != cache_control_capacity) {
						size_t changed = 0;
						// The merged resource memo keeps one state per shader; capacity is informational.
						cache_control_capacity = capacity;
						std::printf("NHL27RESOURCE: cache_capacity=%u changed=%zu\n", capacity, changed);
					}
					unsigned batch = 0;
					if (control >> batch && batch <= 1 && (batch != batch_reads || !batch_policy_reported)) {
						batch_reads = batch != 0;
						batch_policy_reported = true;
						std::printf("NHL27RESOURCE: batch_reads=%u\n", batch);
					}
				}
			}
		}
		last_stats = now;
		std::fflush(stdout);
	}

	// Runtime experiment SrtBatchReads (KYTY_EXPERIMENT_FILE) switches scalar read batching inside
	// one match; compiled shaders and pipelines are kept.
	void ApplyExperiments() {
		constexpr uint32_t Bits = Common::Experiments::SrtBatchReads;
		const uint32_t mask = static_cast<uint32_t>(Common::Experiments::Mask() & Bits);
		if (mask == applied_experiments) return;
		applied_experiments = mask;
		batch_reads = env_batch_reads || (mask & Common::Experiments::SrtBatchReads) != 0;
		std::printf("NHL27RESOURCE: experiments batch_reads=%d\n", batch_reads ? 1 : 0);
	}

	explicit ProgramCache(GraphicContext& graphics): graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
		const char* timing = std::getenv("KYTY_PERF_RESOURCE_TIMING");
		resource_timing = timing != nullptr && timing[0] == '1';
		// Default on: only an explicit KYTY_SRT_BATCH_READS=0 disables it. env_batch_reads (an
		// explicit 1) keeps it on when a KYTY_EXPERIMENT_FILE mask turns the bit off.
		const char* batch = std::getenv("KYTY_SRT_BATCH_READS");
		batch_reads = Common::Experiments::Enabled(Common::Experiments::SrtBatchReads);
		env_batch_reads = batch != nullptr && batch[0] == '1';
		if (resource_timing) {
			if (const char* path = std::getenv("KYTY_RESOURCE_CACHE_CONTROL_FILE")) cache_control_file = path;
		}
	}
	~ProgramCache() {
		StopWorkers();
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	struct LastSource { const ProgramKey* key = nullptr; SourceEntry* source = nullptr; };
	std::array<LastSource, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> last_sources {};
	uint64_t lookups = 0, source_hits = 0, resource_hits = 0, materializations = 0, permutation_hits = 0;
	bool resource_timing = false;
	bool batch_reads = false;
	bool env_batch_reads = false;
	uint32_t applied_experiments = 0;
	bool batch_policy_reported = false;
	std::string cache_control_file;
	unsigned cache_control_capacity = 0;
	std::array<uint64_t, static_cast<size_t>(ResourceMemo::Miss::Count)> resource_misses {};
	std::chrono::steady_clock::time_point last_stats = std::chrono::steady_clock::now();
	GraphicContext&                                             graphics;
    PipelineRecord::RecordFile* record = nullptr;
    const PipelineRecord::Program* replaying = nullptr;
    std::unordered_map<uint64_t, uint32_t> ordinals;
    uint32_t Ordinal(uint64_t id) const {
        const auto found = ordinals.find(id);
        return found == ordinals.end() ? PipelineRecord::NoProgram : found->second;
    }
	// NHL debugging: shader hash -> number of skip warnings printed.
	std::unordered_map<uint64_t, uint32_t>                      failed_shaders;
	std::atomic_uint64_t                                        next_shader_id {0};

	// Graphics programs compiled on worker threads: the resolve thread queues the programs and
	// permutations its draws miss, so the shaders of a new area compile in parallel instead of
	// one after another on the execution thread. Workers only add programs; the execution
	// thread still resolves every draw itself and compiles what they did not add.
	struct CompileJob {
		ProgramKey                                                  key;
		ShaderType                                                  stage = ShaderType::Vertex;
		ShaderParams                                                params;
		std::unique_ptr<ShaderVertexInputInfo>                      vertex;
		std::unique_ptr<ShaderPixelInputInfo>                       pixel;
		std::unique_ptr<ShaderComputeInputInfo>                     compute;
		uint32_t                                                    push_data_cursor = 0;
		std::optional<ShaderRecompiler::IR::ResourceSpecialization> specialization;
		// A program of an earlier session: params.code points here, and it is not stored again.
		std::vector<uint32_t>                                       code;
		std::vector<uint32_t>                                       back_code;
		bool                                                        stored = false;
		// A warm-up record (replay_records): its recorded guest reads derive the specialization.
		const PipelineRecord::Program*                              replay = nullptr;
	};
	static constexpr uint32_t CompileWorkers = 6;

	std::atomic_flag* program_lock = nullptr;
	// Under the program lock: programs a worker is compiling, with their number of jobs.
	std::unordered_map<ProgramKey, uint32_t, ProgramKeyHash> in_flight;
	std::mutex                                     job_mutex;
	std::condition_variable                        job_available;
	std::deque<std::unique_ptr<CompileJob>>        jobs;
	bool                                           stopping = false;
	std::vector<std::jthread>                      workers;

	void LockPrograms() {
		if (!program_lock->test_and_set(std::memory_order_acquire)) {
			return;
		}
		const bool stats = StageStats::Enabled();
		const auto start = stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		do {
			while (program_lock->test(std::memory_order_relaxed)) {
				Common::SpinPause();
			}
		} while (program_lock->test_and_set(std::memory_order_acquire));
		if (stats && StageStats::ProgramLockCounter() != StageStats::CounterCount) {
			StageStats::Add(StageStats::ProgramLockCounter(), start);
		}
	}
	void UnlockPrograms() { program_lock->clear(std::memory_order_release); }

	// Under the program lock (resolve thread).
	template <typename InputInfo>
	void QueueCompile(ShaderType stage, const ShaderParams& params, const InputInfo& input_info,
	                  uint32_t                                            push_data_cursor,
	                  const ShaderRecompiler::IR::ResourceSpecialization* specialization) {
		if (in_flight.contains(lookup_key)) {
			return;
		}
		auto job              = std::make_unique<CompileJob>();
		job->key              = lookup_key;
		job->stage            = stage;
		job->params           = params;
		job->push_data_cursor = push_data_cursor;
		if (specialization != nullptr) {
			job->specialization = *specialization;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			job->vertex = std::make_unique<ShaderVertexInputInfo>(input_info);
		} else {
			job->pixel = std::make_unique<ShaderPixelInputInfo>(input_info);
		}
		in_flight[lookup_key]++;
		{
			std::lock_guard lock(job_mutex);
			jobs.push_back(std::move(job));
		}
		job_available.notify_one();
	}

	// Execution thread, under the program lock: waits without the lock until no worker compiles
	// the program of lookup_key. False when none did.
	bool WaitForCompile() {
		const auto count = in_flight.find(lookup_key);
		if (count == in_flight.end()) {
			return false;
		}
		// Jobs for this program that no worker started yet are taken back: the caller compiles it
		// now instead of waiting behind the queue (a boot queues ~2000 warm-up records).
		{
			std::lock_guard lock(job_mutex);
			const auto taken = std::erase_if(jobs, [&](const auto& job) { return job->key == lookup_key; });
			count->second -= static_cast<uint32_t>(taken);
		}
		if (count->second == 0) {
			in_flight.erase(count);
			return false;
		}
		const auto key = lookup_key;
		do {
			UnlockPrograms();
			// A worker that faults on a GPU-written page of the shader code waits for this
			// thread to read the page back (a menu deadlock on NHL 27).
			GuestGpu::ServiceUrgentCommands();
			std::this_thread::yield();
			LockPrograms();
		} while (in_flight.contains(key));
		return true;
	}

	template <typename InputInfo>
	void RunCompileJob(CompileJob& job, InputInfo& input_info) {
		std::optional<ShaderRecompiler::IR::ResourceSpecialization> stored_specialization;
		std::optional<Permutation>                        permutation;
		std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
		try {
			Common::SoftExitScope soft_exit(true);
			const auto [options, stage_name] =
			    MakeCompileOptions(job.stage, job.params, input_info);
			// SWAPPC programs link the functions they call from guest memory: the execution
			// thread compiles them.
			auto source = ShaderRecompiler::PrepareShaderSource(job.params.code, options);
			if (source.call) {
				throw Common::SoftExitError("SWAPPC program");
			}
			auto translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
			if (translated.build_failed) {
				throw Common::SoftExitError("shader CFG build failed");
			}
			plan                = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			auto specialization = job.specialization;
			if (!specialization) {
				GuestReadLog                           reads;
				ShaderRecompiler::IR::ResourceSnapshot resources;
				specialization.emplace();
				// Warm-up records run before the game is loaded: they read their recorded words.
				auto* replay = const_cast<PipelineRecord::Program*>(job.replay);
				const ShaderRecompiler::IR::SrtRuntime runtime {
				    .user_data =
				        std::span(job.params.user_data).first(job.params.user_data_count),
				    .shader_base                = job.params.Base(),
				    .read_memory                = replay != nullptr ? ReplayRawWords : ReadAheadPlain,
				    .userdata = replay != nullptr ? static_cast<void*>(replay) : &reads,
				    .read_specialization_memory =
				        replay != nullptr ? ReplayCleanWords : ReadAheadStrict,
				};
				if (!ShaderRecompiler::IR::MaterializeResources(*plan, runtime, resources,
				                                                *specialization)) {
					specialization.reset();
				}
			}
			if (specialization) {
				permutation =
				    CompilePermutation(stage_name, options, std::move(translated),
				                       std::move(*specialization), job.push_data_cursor, {});
			}
		} catch (const Common::SoftExitError&) {
			// The execution thread compiles the program itself and reports the error.
			permutation.reset();
		}
		LockPrograms();
		if (permutation) {
			auto entry = programs.find(job.key);
			if (entry == programs.end()) {
				entry = programs.try_emplace(job.key, std::move(*plan), nullptr).first;
			}
			auto&       permutations = entry->second.permutations;
			const auto& layout       = permutation->program.bindings;
			const bool  known = std::ranges::any_of(permutations, [&](const Permutation& other) {
				return other.specialization == permutation->specialization &&
				       other.program.bindings.push_data_start_dword == layout.push_data_start_dword;
			});
			if (known) {
				graphics.device.destroyShaderModule(permutation->handle.module, nullptr);
			} else {
				if (store != nullptr && !job.stored) {
					stored_specialization = permutation->specialization;
				}
				permutations.push_back(std::move(*permutation));
			}
		}
		if (const auto job_count = in_flight.find(job.key); --job_count->second == 0) {
			in_flight.erase(job_count);
		}
		UnlockPrograms();
		if (stored_specialization) {
			Store(job.key, job.params, input_info, job.push_data_cursor, *stored_specialization);
		}
	}

	void CompileWorker(const std::stop_token& stop) {
		KYTY_PROFILER_THREAD("Thread_ShaderCompile");
		// Background compiles (store, warm-up records) yield to the game's own threads.
		Common::Thread::LowerCurrentPriority();
		for (;;) {
			std::unique_ptr<CompileJob> job;
			{
				std::unique_lock lock(job_mutex);
				job_available.wait(lock, [&] { return stopping || !jobs.empty(); });
				if (stopping) {
					return;
				}
				job = std::move(jobs.front());
				jobs.pop_front();
			}
			if (job->vertex != nullptr) {
				RunCompileJob(*job, *job->vertex);
			} else if (job->pixel != nullptr) {
				RunCompileJob(*job, *job->pixel);
			} else {
				RunCompileJob(*job, *job->compute);
			}
		}
	}

	std::unique_ptr<ProgramStore> store;

	template <typename InputInfo>
	void Store(const ProgramKey& key, const ShaderParams& params, const InputInfo& input_info,
	           uint32_t                                            push_data_cursor,
	           const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
		if (store == nullptr || replaying != nullptr) {
			return;
		}
		// A permutation loaded from the store is not stored again (store_specializations).
		if (const auto stored = store_specializations.find(key);
		    stored != store_specializations.end() &&
		    std::ranges::find(stored->second, specialization) != stored->second.end()) {
			return;
		}
		StoredProgram program;
		program.stage           = key.stage;
		program.hash            = key.hash;
		program.user_data_count = key.user_data_count;
		program.code_size       = key.code_size;
		program.static_state    = key.static_state;
		program.code.assign(params.code.begin(), params.code.end());
		program.back_code.assign(params.back_code.begin(), params.back_code.end());
		program.user_data        = params.user_data;
		program.push_data_cursor = push_data_cursor;
		program.specialization   = specialization;
		auto info                = input_info;
		info.stage               = {};
		program.input_info.resize(sizeof(info));
		std::memcpy(program.input_info.data(), &info, sizeof(info));
		store->Append(program);
	}

	template <typename InputInfo>
	static std::unique_ptr<InputInfo> LoadInputInfo(const std::vector<uint8_t>& bytes) {
		if (bytes.size() != sizeof(InputInfo)) {
			return nullptr;
		}
		auto info = std::make_unique<InputInfo>();
		std::memcpy(info.get(), bytes.data(), sizeof(InputInfo));
		info->stage = {};
		return info;
	}

	// Queues the programs of earlier sessions to the workers (before any draw runs).
	void LoadStore(const std::filesystem::path& path) {
		store = std::make_unique<ProgramStore>();
		auto programs = store->Open(path);
		size_t queued = 0;
		for (auto& program: programs) {
			auto job   = std::make_unique<CompileJob>();
			job->key   = {.stage           = program.stage,
			              .hash            = program.hash,
			              .user_data_count = program.user_data_count,
			              .code_size       = program.code_size,
			              .static_state    = std::move(program.static_state)};
			job->stage = program.stage;
			switch (program.stage) {
				case ShaderType::Pixel:
					job->pixel = LoadInputInfo<ShaderPixelInputInfo>(program.input_info);
					break;
				case ShaderType::Compute:
					job->compute = LoadInputInfo<ShaderComputeInputInfo>(program.input_info);
					break;
				case ShaderType::Vertex:
				case ShaderType::Mesh:
				case ShaderType::Local:
				case ShaderType::TessellationControl:
				case ShaderType::TessellationEvaluation:
					job->vertex = LoadInputInfo<ShaderVertexInputInfo>(program.input_info);
					break;
				default: break;
			}
			if (job->vertex == nullptr && job->pixel == nullptr && job->compute == nullptr) {
				continue;
			}
			// The input info must give the stored key (and be in range for building it).
			if ((job->vertex != nullptr &&
			     (job->vertex->resources_num < 0 ||
			      job->vertex->resources_num > ShaderVertexInputInfo::RES_MAX ||
			      job->vertex->buffers_num < 0 ||
			      job->vertex->buffers_num > ShaderVertexInputInfo::RES_MAX)) ||
			    (job->pixel != nullptr &&
			     job->pixel->input_num > std::size(job->pixel->interpolator_settings))) {
				continue;
			}
			std::vector<uint32_t> static_state;
			if (job->vertex != nullptr) {
				BuildStageStaticKey(*job->vertex, static_state);
			} else if (job->pixel != nullptr) {
				BuildStageStaticKey(*job->pixel, static_state);
			} else {
				BuildStageStaticKey(*job->compute, static_state);
			}
			if (static_state != job->key.static_state) {
				continue;
			}
			job->code                   = std::move(program.code);
			job->back_code              = std::move(program.back_code);
			job->params.code            = job->code;
			job->params.back_code       = job->back_code;
			job->params.user_data       = program.user_data;
			job->params.user_data_count = program.user_data_count;
			job->params.hash            = program.hash;
			job->push_data_cursor       = program.push_data_cursor;
			job->specialization         = std::move(program.specialization);
			job->stored                 = true;
			store_specializations[job->key].push_back(*job->specialization);
			LockPrograms();
			in_flight[job->key]++;
			UnlockPrograms();
			{
				std::lock_guard lock(job_mutex);
				jobs.push_back(std::move(job));
			}
			queued++;
		}
		job_available.notify_all();
		PipelineCacheLog("Shader program store: {} programs from {} queued for compilation", queued,
		                 Common::PathToString(path));
	}

	// Warm-up records of earlier sessions, replayed on the workers (QueueRecords).
	PipelineRecord::Records replay_records;
	// What the store and the record held at startup (read-only afterwards): a program compiled
	// again because its job was taken back (WaitForCompile) is not stored or recorded a second
	// time, so neither file grows with duplicates every boot. The store compares permutations (a
	// new specialization of a stored program is still stored); the record compares keys.
	std::unordered_map<ProgramKey, std::vector<ShaderRecompiler::IR::ResourceSpecialization>,
	                   ProgramKeyHash>
	                                               store_specializations;
	std::unordered_set<ProgramKey, ProgramKeyHash> record_keys;

	// Background warm-up: the recorded programs compile on the workers with their recorded guest
	// reads, so the window opens without waiting ~2.5 min for them; a draw that needs one first
	// waits for its job (WaitForCompile). Keys the store queued are skipped. Pipelines are created
	// on first use, which the driver cache keeps short. KYTY_SHADER_WARMUP_SYNC=1: the old
	// blocking warm-up (WarmUp).
	void QueueRecords() {
		if (replay_records.programs.empty()) {
			return;
		}
		size_t queued   = 0;
		size_t in_store = 0;
		LockPrograms();
		for (const auto& source: replay_records.programs) {
			if (source.user_data.size() > ShaderParams {}.user_data.size()) {
				continue;
			}
			auto job   = std::make_unique<CompileJob>();
			job->stage = source.stage;
			job->key   = {.stage           = source.stage,
			              .hash            = source.hash,
			              .user_data_count = static_cast<uint32_t>(source.user_data.size()),
			              .code_size       = static_cast<uint32_t>(source.code.size())};
			switch (source.stage) {
				case ShaderType::Vertex:
					if (source.vertex.native_draw.Enabled() &&
					    !graphics.native_draw_parameters_enabled) {
						continue;
					}
					job->vertex        = std::make_unique<ShaderVertexInputInfo>(source.vertex);
					job->vertex->stage = {};
					BuildStageStaticKey(*job->vertex, job->key.static_state);
					break;
				case ShaderType::Pixel: {
					job->pixel        = std::make_unique<ShaderPixelInputInfo>(source.pixel);
					job->pixel->stage = {};
					const char* quad  = std::getenv("KYTY_PIXEL_QUAD_CORRECT");
					job->pixel->pixel_quad_correct =
					    graphics.pixel_quad_supported && quad != nullptr && quad[0] == '1';
					BuildStageStaticKey(*job->pixel, job->key.static_state);
					break;
				}
				case ShaderType::Compute:
					job->compute        = std::make_unique<ShaderComputeInputInfo>(source.compute);
					job->compute->stage = {};
					job->compute->host_subgroup_size = graphics.SupportsComputeWave64() ? 64u : 32u;
					job->compute->lds_storage =
					    job->compute->lds_size_dwords * 4u >
					    graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
					BuildStageStaticKey(*job->compute, job->key.static_state);
					break;
				default: continue;
			}
			record_keys.insert(job->key);
			if (store_specializations.contains(job->key)) {
				in_store++;
				continue;
			}
			job->params.code            = source.code;
			job->params.back_code       = source.back_code;
			job->params.base            = source.shader_base;
			job->params.hash            = source.hash;
			job->params.user_data_count = static_cast<uint32_t>(source.user_data.size());
			std::ranges::copy(source.user_data, job->params.user_data.begin());
			job->push_data_cursor = source.push_data_cursor;
			job->replay           = &source;
			in_flight[job->key]++;
			{
				std::lock_guard lock(job_mutex);
				jobs.push_back(std::move(job));
			}
			queued++;
		}
		UnlockPrograms();
		job_available.notify_all();
		PipelineCacheLog("Shader warm-up: {} recorded programs queued for background compilation "
		                 "({} already queued from the store)",
		                 queued, in_store);
	}

	void StartWorkers(std::atomic_flag* lock) {
		program_lock = lock;
		for (uint32_t i = 0; i < CompileWorkers; i++) {
			workers.emplace_back([this](const std::stop_token& stop) { CompileWorker(stop); });
		}
	}

	void StopWorkers() {
		{
			std::lock_guard lock(job_mutex);
			stopping = true;
		}
		job_available.notify_all();
		workers.clear();
	}
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// LoadStore takes the program lock, so it must be set before the store is read.
	m_program_cache->program_lock = &m_program_lock;
	InitializeDriverCache();
	InitializeShaderRecord();
	const char* pregen = std::getenv("KYTY_SHADER_PREGEN");
	if (const auto title_id = PipelineCacheTitleId();
	    !title_id.empty() && KYTY_BUILD == KYTY_BUILD_RELEASE && Config::PreGenEnabled() &&
	    (pregen == nullptr || pregen[0] != '0')) {
		// One store per build: the records copy the layouts of the stage input structures, and
		// what a program key means may change between builds. A build with local changes uses the
		// time pipelineCache.cpp (which includes those structures) was compiled.
		const std::string_view hash = KYTY_GIT_HASH;
		auto build = std::string(std::string_view(KYTY_GIT_REVISION).substr(0, 12));
		if (hash.ends_with("-dirty")) {
			build += fmt::format("-{:08x}", static_cast<uint32_t>(XXH3_64bits(
			                                    __DATE__ __TIME__, sizeof(__DATE__ __TIME__))));
		}
		const auto directory = std::filesystem::path("_PipelineCache");
		const auto name      = fmt::format("{}.{}.programs", title_id, build);
		// Builds that share the directory (a debug and a release folder linked to one cache) keep
		// their own stores; only the oldest stores beyond a few recent builds are removed.
		constexpr size_t KeptStores = 4;
		std::error_code  error;
		std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> stores;
		for (const auto& file: std::filesystem::directory_iterator(directory, error)) {
			const auto other = file.path().filename().string();
			if (other != name && other.starts_with(title_id + ".") && other.ends_with(".programs")) {
				stores.emplace_back(file.last_write_time(error), file.path());
			}
		}
		if (stores.size() >= KeptStores) {
			std::ranges::sort(stores, std::greater {}, [](const auto& store) { return store.first; });
			for (size_t index = KeptStores - 1; index < stores.size(); index++) {
				std::filesystem::remove(stores[index].second, error);
			}
		}
		m_program_cache->LoadStore(directory / name);
	}
	m_program_cache->QueueRecords();
	m_program_cache->StartWorkers(&m_program_lock);
}

PipelineCache::~PipelineCache() {
	m_program_cache->StopWorkers();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= 1024ull * 1024ull * 1024ull) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
	m_save_thread = std::jthread([this](const std::stop_token& stop) {
		constexpr auto Period = std::chrono::seconds(60);
		auto           next   = std::chrono::steady_clock::now() + Period;
		while (!stop.stop_requested()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			if (std::chrono::steady_clock::now() < next) {
				continue;
			}
			next = std::chrono::steady_clock::now() + Period;
			if (m_unsaved_pipelines.exchange(0) != 0) {
				WriteDriverCache();
			}
		}
	});
}

void PipelineCache::Save() {
	if (m_save_thread.joinable()) {
		m_save_thread.request_stop();
		m_save_thread.join();
	}
	if (m_driver_cache == nullptr) {
		return;
	}
	WriteDriverCache();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::WriteDriverCache() {
	// Pipeline creation may continue meanwhile: the driver synchronizes the cache object.
	Common::LockGuard lock(m_write_mutex);

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::InitializeShaderRecord() {
	const char* flag = std::getenv("KYTY_SHADER_PREGEN");
	if (!Config::PreGenEnabled() || (flag != nullptr && flag[0] == '0')) {
		PipelineCacheLog("Shader warm-up: disabled (--pre-gen false)");
		return;
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	const auto path = std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
	PipelineRecord::Records records;
	m_record = std::make_unique<PipelineRecord::RecordFile>();
	if (!m_record->Open(path, records)) {
		PipelineCacheLog("Shader warm-up: cannot write {}; new shaders are not recorded",
		                 Common::PathToString(path));
	}
	m_program_cache->record = m_record.get();
	// KYTY_SHADER_WARMUP_SYNC=1: compile the records here, before the window opens. Otherwise the
	// constructor queues them to the workers after the store (ProgramCache::QueueRecords).
	static const bool sync = [] {
		const char* value = std::getenv("KYTY_SHADER_WARMUP_SYNC");
		return value != nullptr && value[0] == '1';
	}();
	if (!sync) {
		m_program_cache->replay_records = std::move(records);
		return;
	}
	WarmUp(records);
}


void PipelineCache::WarmUp(PipelineRecord::Records& records) {
    if (records.programs.empty()) {
        PipelineCacheLog("Shader warm-up: no records yet; recording during play");
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    // Shared ProgramCache/compiler state is serialized; do not race translation or insertion.
    std::vector<ShaderProgram> handles(records.programs.size());
    std::vector<ShaderRecompiler::IR::ResourceSnapshot> snapshots(records.programs.size());
    for (size_t index = 0; index < records.programs.size(); ++index) {
        auto& source = records.programs[index];
        if (source.user_data.size() > ShaderParams{}.user_data.size()) continue;
        if (source.stage == ShaderType::Vertex && source.vertex.native_draw.Enabled() &&
            !m_graphics.native_draw_parameters_enabled) continue;
        if (source.stage == ShaderType::Pixel) {
            const char* quad = std::getenv("KYTY_PIXEL_QUAD_CORRECT");
            source.pixel.pixel_quad_correct = m_graphics.pixel_quad_supported &&
                quad != nullptr && quad[0] == '1';
        }
        if (source.stage == ShaderType::Compute) {
            source.compute.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
            source.compute.lds_storage = source.compute.lds_size_dwords * 4u >
                m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
        }
        ShaderParams params;
        params.code = source.code; params.back_code = source.back_code;
        params.base = source.shader_base; params.hash = source.hash;
        params.user_data_count = static_cast<uint32_t>(source.user_data.size());
        std::copy(source.user_data.begin(), source.user_data.end(), params.user_data.begin());
        auto cursor = source.push_data_cursor;
        m_program_cache->replaying = &source;
        switch (source.stage) {
            case ShaderType::Vertex: handles[index] = m_program_cache->Get(params, source.vertex, cursor); break;
            case ShaderType::Pixel: handles[index] = m_program_cache->Get(params, source.pixel, cursor); break;
            case ShaderType::Compute: handles[index] = m_program_cache->Get(params, source.compute, cursor); break;
            default: break;
        }
        m_program_cache->replaying = nullptr;
        if (handles[index]) {
            const ShaderStageRuntime* runtime = nullptr;
            switch (source.stage) {
                case ShaderType::Vertex: runtime = &source.vertex.stage; break;
                case ShaderType::Pixel: runtime = &source.pixel.stage; break;
                case ShaderType::Compute: runtime = &source.compute.stage; break;
                default: break;
            }
            if (runtime != nullptr && runtime->resources != nullptr)
                snapshots[index] = *runtime->resources;
        }
        if (handles[index]) m_program_cache->ordinals[handles[index].id] = static_cast<uint32_t>(index);
    }
    // Replay records omit runtime pointers. Rebind both parts of the stage after
    // all permutations have been inserted, since insertion can move their programs.
    std::unordered_map<uint64_t, ShaderStageRuntime> compiled;
    for (const auto& [key, entry]: m_program_cache->programs)
        for (const auto& permutation: entry.permutations)
            compiled[permutation.handle.id] = {.program = &permutation.program,
                                              .resources = &entry.resources};
    size_t graphics_count = 0, compute_count = 0;
    for (auto& record: records.graphics) {
        const auto vertex = handles[record.vertex_program];
        const bool ps_active = record.pixel_program != PipelineRecord::NoProgram;
        const auto pixel = ps_active ? handles[record.pixel_program] : ShaderProgram{};
        if (!vertex || (ps_active && !pixel)) continue;
        GraphicsPipelineKey key;
        key.rendering = record.rendering; key.vertex_shader_ids[0] = vertex.id;
        key.ps_shader_id = pixel.id; key.vertex_input = record.vertex_input;
        key.static_params = record.static_params;
        if (m_graphics_pipelines.contains(key)) continue;
        GraphicsPrograms programs; programs.vertex[0] = vertex; programs.pixel = pixel;
        record.vertex.stage = compiled.at(vertex.id);
        if (ps_active) record.pixel.stage = compiled.at(pixel.id);
        auto pipeline = std::make_unique<Pipeline>();
        CreatePipelineInternal(m_graphics, *pipeline, key.rendering, key.vertex_input,
            std::span<const ShaderVertexInputInfo>(&record.vertex, 1),
            ps_active ? &record.pixel : nullptr, programs, key.static_params, m_driver_cache);
        m_graphics_pipelines.emplace(std::move(key), std::move(pipeline)); ++graphics_count;
    }
    for (auto& record: records.compute) {
        const auto program = handles[record.program];
        if (!program || m_compute_pipelines.contains(program.id)) continue;
        record.compute.stage = compiled.at(program.id);
        record.compute.host_subgroup_size = records.programs[record.program].compute.host_subgroup_size;
        record.compute.lds_storage = records.programs[record.program].compute.lds_storage;
        auto pipeline = std::make_unique<Pipeline>();
        CreatePipelineInternal(m_graphics, *pipeline, record.compute, program.module, m_driver_cache);
        m_compute_pipelines.emplace(program.id, std::move(pipeline)); ++compute_count;
    }
    if (m_driver_cache != nullptr) {
        WriteDriverCache();
    }
    PipelineCacheLog("Shader warm-up: {} programs, {} graphics and {} compute pipelines in {:.1f}s",
        std::ranges::count_if(handles, [](const auto& handle) { return bool(handle); }),
        graphics_count, compute_count,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

void PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey&             key,
                                           std::span<const ShaderVertexInputInfo> vertex_info,
                                           const ShaderPixelInputInfo*            ps_input_info) {
	if (m_record == nullptr || !m_record->IsOpen() || vertex_info.size() != 1 ||
	    vertex_info[0].logical_stage != ShaderType::Vertex) {
		return;
	}
	PipelineRecord::GraphicsPipeline record;
	record.vertex_program = m_program_cache->Ordinal(key.vertex_shader_ids[0]);
	record.pixel_program  = ps_input_info != nullptr ? m_program_cache->Ordinal(key.ps_shader_id)
	                                                 : PipelineRecord::NoProgram;
	if (record.vertex_program == PipelineRecord::NoProgram ||
	    (ps_input_info != nullptr && record.pixel_program == PipelineRecord::NoProgram)) {
		return;
	}
	record.rendering     = key.rendering;
	record.vertex_input  = key.vertex_input;
	record.static_params = key.static_params;
	record.vertex        = vertex_info[0];
	if (ps_input_info != nullptr) {
		record.pixel = *ps_input_info;
	}
	m_record->AppendGraphicsPipeline(record);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info, const NativeDrawShaderInfo& native_draw, ProgramResolution* ahead) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_graphics_programs_ns,
	                                 FrameBreakdown::DetailEnabled());
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	if (ahead != nullptr && tess_active) {
		ahead->reads.Fail();
		return {};
	}
	// Ahead of execution, the vertex fetch tables are read through the resolution's log.
	const ShaderGuestReaderScope reader(ahead != nullptr ? ReadAheadShaderState : nullptr,
	                                    ahead != nullptr ? &ahead->reads : nullptr);
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	if (vertex_info[0].logical_stage == ShaderType::Vertex) vertex_info[0].native_draw = native_draw;
	if (ahead != nullptr && ahead->reads.Failed()) {
		return {};
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		static const bool quad_enabled = [] {
			const char* flag = std::getenv("KYTY_PIXEL_QUAD_CORRECT");
			return flag != nullptr && flag[0] == '1';
		}();
		pixel_info.pixel_quad_correct = quad_enabled && m_graphics.pixel_quad_supported;
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms       result;
	const ProgramLockGuard lock(m_program_lock);
	if (ahead != nullptr) {
		if (pixel_active) {
			result.pixel =
			    m_program_cache->GetAhead(pixel_params, pixel_info, push_data_cursor, *ahead);
		}
		if (!ahead->reads.Failed()) {
			result.vertex[0] = m_program_cache->GetAhead(vertex_params[0], vertex_info[0],
			                                             push_data_cursor, *ahead);
		}
		return ahead->reads.Failed() ? GraphicsPrograms {} : result;
	}
	bool build_failed = false;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
		build_failed |= !result.pixel;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
		build_failed |= !result.vertex[i];
	}
	// NHL debugging: an empty vertex[0] tells the draw path to skip the draw.
	return build_failed ? GraphicsPrograms {} : result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info,
                                               ProgramResolution*          ahead) {
	FrameBreakdown::Timer prep_timer(FrameBreakdown::g_counters.prep_compute_program_ns,
	                                 FrameBreakdown::DetailEnabled());
	const ShaderGuestReaderScope reader(ahead != nullptr ? ReadAheadShaderState : nullptr,
	                                    ahead != nullptr ? &ahead->reads : nullptr);
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	if (ahead != nullptr && ahead->reads.Failed()) {
		return {};
	}
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	const ProgramLockGuard lock(m_program_lock);
	if (ahead != nullptr) {
		return m_program_cache->GetAhead(params, input_info, push_data_cursor, *ahead);
	}
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

ShaderProgram PipelineCache::GetComputeProgramAhead(const HW::ComputeShaderInfo& regs,
                                                    const HW::ShaderRegisters& sh,
                                                    ShaderComputeInputInfo& input_info, ProgramResolution& ahead) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const ShaderGuestReaderScope reader(ReadAheadShaderState, &ahead.reads);
	const auto params = PrepareProgram(regs, sh, input_info);
	if (ahead.reads.Failed()) {
		return {};
	}
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t               push_data_cursor = 0;
	const ProgramLockGuard lock(m_program_lock);
	auto program = m_program_cache->GetAhead(params, input_info, push_data_cursor, ahead);
	return ahead.reads.Failed() ? ShaderProgram {} : program;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

static bool PipelineMemoVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAWPREP_VERIFY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	FrameBreakdown::Timer key_timer(FrameBreakdown::g_pl.key_ns, FrameBreakdown::Verbose());
	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	key_timer.Stop();
	{
		FrameBreakdown::Timer lookup_timer(FrameBreakdown::g_pl.lookup_ns, FrameBreakdown::Verbose());
		FrameBreakdown::VAdd(FrameBreakdown::g_pl.lookups);
		// A key equal to the previous call's key maps to the previous call's pipeline (the map never changes an
		// existing entry), so the hash + lookup can be skipped. The compare also feeds the key_repeat counter.
		const bool memo_on = Common::RuntimeFlags::Current().pipeline_memo;
		if (m_last_pipeline != nullptr && (memo_on || FrameBreakdown::Verbose()) && key == m_last_key) {
			FrameBreakdown::VAdd(FrameBreakdown::g_pl.key_repeats);
			if (memo_on) {
				static uint32_t verify_tick = 0;
				if (PipelineMemoVerifyEnabled() && (++verify_tick & 63u) == 0) {
					const auto full = m_graphics_pipelines.find(key);
					FrameBreakdown::VAdd(FrameBreakdown::g_pl.memo_verify);
					if (full == m_graphics_pipelines.end() || full->second.get() != m_last_pipeline) {
						FrameBreakdown::VAdd(FrameBreakdown::g_pl.memo_mismatch);
						std::fprintf(stderr, "NHL27PIPELINE memo mismatch: graphics key (vs=%" PRIu64 " ps=%" PRIu64 ")\n",
						             vs_id, ps_id);
						std::fflush(stderr);
						if (full != m_graphics_pipelines.end()) {
							m_last_pipeline = full->second.get();
						}
					}
				}
				return *m_last_pipeline;
			}
		}
		if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
			if (memo_on || FrameBreakdown::Verbose()) {
				m_last_key      = key;
				m_last_pipeline = iter->second.get();
			}
			return *iter->second;
		}
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);
	m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	RecordGraphicsPipeline(key, vertex_info, ps_input_info);
	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	FrameBreakdown::VAdd(FrameBreakdown::g_pl.cs_lookups);
	if (m_last_compute_id == compute_program.id) {
		FrameBreakdown::VAdd(FrameBreakdown::g_pl.cs_repeats);
	}
	m_last_compute_id = compute_program.id;
	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);
	}
	m_unsaved_pipelines.fetch_add(1, std::memory_order_relaxed);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

if (m_record != nullptr && m_record->IsOpen() && m_program_cache->replaying == nullptr) {
        const auto ordinal = m_program_cache->Ordinal(compute_program.id);
        if (ordinal != PipelineRecord::NoProgram) m_record->AppendComputePipeline({ordinal, input_info});
    }
    auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);

	return *iter->second;
}
} // namespace Libs::Graphics
