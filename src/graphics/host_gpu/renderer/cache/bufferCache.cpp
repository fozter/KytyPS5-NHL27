#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

/// Zero BDA mappings and fault bits once, before recording their first GPU use.
void BufferCache::EnsureBdaStorageInitialized() {
	if (m_bda_storage_initialized) return;
	// Device-local allocations may contain recycled data. Zero is the missing-page
	// sentinel; stale addresses must never be interpreted as live mappings.
	// Do this on first GPU use, since construction has no recording command buffer.
	m_bda_pagetable_buffer.Fill(0, m_bda_pagetable_buffer.Size(), 0);
	auto* faults = m_fault_manager.GetFaultBuffer();
	faults->Fill(0, faults->Size(), 0);
	m_bda_storage_initialized = true;
}

/// Return the page table after scheduling its required initial clear.
Buffer* BufferCache::GetBdaPageTableBuffer() noexcept {
	EnsureBdaStorageInitialized();
	return &m_bda_pagetable_buffer;
}

/// Return the fault bitset after scheduling its required initial clear.
Buffer* BufferCache::GetFaultBuffer() noexcept {
	EnsureBdaStorageInitialized();
	return m_fault_manager.GetFaultBuffer();
}

/// Initialize BDA storage before publishing a new guest-to-device mapping.
void BufferCache::Register(BufferId id) {
	EnsureBdaStorageInitialized();
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		m_memory_tracker.QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	// Diagnostic (uncommitted): KYTY_SYNC_WATCH=<hex address in a 64 KiB page> prints, for the first
	// 400 downloads that page triggers, the request, the PM4 opcode and every GPU-modified range
	// downloaded with the shader that last wrote it: a GPU-written flag vs false sharing in its page.
	static const uint64_t watch_page = [] {
		const char* value = std::getenv("KYTY_SYNC_WATCH");
		return value != nullptr ? std::strtoull(value, nullptr, 16) >> 16u : UINT64_MAX;
	}();
	static uint32_t watch_printed = 0;
	if ((vaddr >> 16u) == watch_page && watch_printed < 400) {
		watch_printed++;
		std::string ranges;
		for (const auto& copy: copies) {
			const auto start  = buffer_address + copy.srcOffset;
			uint64_t   writer = 0;
			auto       it     = g_gpu_write_ranges.upper_bound(start);
			for (int steps = 0; steps < 64 && it != g_gpu_write_ranges.begin(); steps++) {
				--it;
				if (it->first <= start && start < it->first + it->second.first) {
					writer = it->second.second;
					break;
				}
			}
			char part[96];
			std::snprintf(part, sizeof(part), " [0x%llx+0x%llx cs %016llx]",
			              static_cast<unsigned long long>(start),
			              static_cast<unsigned long long>(copy.size),
			              static_cast<unsigned long long>(writer));
			ranges += part;
		}
		std::fprintf(stderr, "[syncwatch] request 0x%llx+0x%llx op %x gpu-thread %d ->%s\n",
		             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
		             g_sync_cp_op, GuestGpu::IsGpuThread() ? 1 : 0, ranges.c_str());
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBackingSkipHoles(buffer_address + copy.srcOffset,
			                                               mapped + (copy.dstOffset - offset),
			                                               copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	g_diag_buffer_cache = this;
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
}

// Perf (uncommitted): download GPU-written bytes on the readback queue, waiting
// only for the submission (timeline tick) that wrote them. The old path drains
// the whole command pipeline via the scheduler; this keeps the graphics queue
// running in parallel and the in-progress command buffer unflushed.
bool BufferCache::DownloadViaReadbackQueue(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                           bool is_write, uint64_t wait_tick) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address,
	                                                             uint64_t bytes) noexcept {
		m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                       "buffer download");
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			copies.emplace_back(start - buffer_address, total_size, end - start);
			// Keep packed ranges on separate cache lines, as in shadPS4.
			total_size += Common::AlignUp(end - start, 64);
		});
		m_gpu_modified_ranges.Subtract(address, bytes);
	});
	if (copies.empty()) {
		return false;
	}

	// Per-download temporary staging: the shared download stream belongs to the
	// scheduler's command recording and cannot be committed from here.
	auto        temporary = std::make_unique<Buffer>(m_graphics, m_scheduler,
              MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, total_size);
	uint8_t*    mapped    = temporary->Mapped().data();

	vk::CommandBuffer command {};
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_graphics.readback_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	EXIT_NOT_IMPLEMENTED(m_graphics.device.allocateCommandBuffers(&allocate, &command) !=
	                     vk::Result::eSuccess);

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	EXIT_NOT_IMPLEMENTED(command.begin(&begin_info) != vk::Result::eSuccess);

	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                         vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                         nullptr);
	command.copyBuffer(buffer.Handle(), temporary->Handle(), static_cast<uint32_t>(copies.size()),
	                   copies.data());

	auto after   = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = temporary->Handle();
	after.offset       = 0;
	after.size         = total_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                         vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
	                         nullptr);
	EXIT_NOT_IMPLEMENTED(command.end() != vk::Result::eSuccess);

	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount = 1;
	const uint64_t wait_value        = wait_tick;
	timeline.pWaitSemaphoreValues    = &wait_value;
	vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
	auto&                 master      = m_scheduler.GetMasterSemaphore();
	vk::Semaphore          semaphore   = master.Handle();
	vk::SubmitInfo        submit_info {};
	submit_info.pNext              = &timeline;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores    = &semaphore;
	submit_info.pWaitDstStageMask  = &wait_stage;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &command;

	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		EXIT_NOT_IMPLEMENTED(m_graphics.readback_queue.submit(1, &submit_info, nullptr) !=
		                     vk::Result::eSuccess);
	}
	// The readback queue is used only by this single sequential download path,
	// so a queue wait equals waiting for this submission's copy.
	EXIT_NOT_IMPLEMENTED(m_graphics.readback_queue.waitIdle() != vk::Result::eSuccess);

	temporary->Invalidate(0, total_size);
	for (const auto& copy: copies) {
		Libs::LibKernel::Memory::WriteBackingSkipHoles(buffer_address + copy.srcOffset,
		                                               mapped + copy.dstOffset, copy.size);
	}
	m_graphics.device.freeCommandBuffers(m_graphics.readback_pool, 1, &command);
	return true;
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// Diagnostic (uncommitted): readback latency accounting on both paths.
	static uint64_t diag_calls = 0;
	static double   diag_wait  = 0.0;
	static double   diag_max   = 0.0;
	static uint64_t diag_fast  = 0;
	const auto      diag_t0    = std::chrono::steady_clock::now();
	const auto      diag_done = [&](const char* path) {
		const double wait_ms =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - diag_t0)
		        .count();
		++diag_calls;
		diag_wait += wait_ms;
		if (wait_ms > diag_max) {
			diag_max = wait_ms;
		}
		if (wait_ms >= 0.5) {
			std::fprintf(stderr,
			             "[gpusyncdiag] %s wait=%.1fms addr=0x%016" PRIx64 " size=0x%" PRIx64
			             " write=%d\n",
			             path, wait_ms, vaddr, size, static_cast<int>(is_write));
			std::fflush(stderr);
		}
		if ((diag_calls & 0xffu) == 0) {
			std::fprintf(stderr,
			             "[gpusyncdiag] total: %llu calls (%llu fast), %.0f ms waited, max %.1f ms\n",
			             static_cast<unsigned long long>(diag_calls),
			             static_cast<unsigned long long>(diag_fast), diag_wait, diag_max);
			std::fflush(stderr);
		}
	};

	// Fast path: every recorded GPU write has already been submitted (a later
	// tick exists than the last write marking), and the range actually holds
	// GPU-dirty bytes. Download on the readback queue waiting only for the
	// last submitted tick, leaving the in-progress command buffer unflushed
	// and the graphics queue running in parallel.
	const uint64_t write_tick   = m_last_write_tick.load(std::memory_order_acquire);
	const uint64_t current_tick = m_scheduler.CurrentTick();
	static const bool diag_fast_enabled = std::getenv("KYTY_READBACK_FAST") != nullptr;
	// KYTY_READBACK_FAST2 (experiment, uncommitted): the deadlock-free variant. Never on the GPU
	// thread (only it submits, so waiting there is circular: the 2026-10-18 boot stall), and only
	// when the last buffer write is at least two submissions old, so the command that performs it
	// is in submitted work even if a submit happened between its marking and its recording (the
	// race that returned stale data). It then waits on the last SUBMITTED tick, never on the one
	// being recorded. KYTY_AB=1 alternates it with the drain every 128 GC ticks.
	static const bool fast2_enabled = std::getenv("KYTY_READBACK_FAST2") != nullptr;
	static const bool fast2_ab      = std::getenv("KYTY_AB") != nullptr;
	const bool        fast2_active  = fast2_enabled && (!fast2_ab || ((m_gc_tick / 128) % 2) == 0);
	if (fast2_active && m_graphics.readback_queue != nullptr && !GuestGpu::IsGpuThread() &&
	    write_tick != 0 && write_tick + 2 <= current_tick) {
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		if ((!is_write || IsRegionRegistered(vaddr, size)) &&
		    DownloadViaReadbackQueue(buffer, vaddr, size, is_write, current_tick - 1)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			++diag_fast;
			diag_done("fast2");
			return;
		}
	}
	if (diag_fast_enabled && m_graphics.readback_queue != nullptr && write_tick != 0 &&
	    write_tick < current_tick) {
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
		// Wait on the CURRENT (last submitted) tick, not the mark-epoch tick:
		// the buffer that recorded the write submits at some tick in
		// (write_tick, current_tick], so current_tick is the safe upper bound.
		// Ranges with no GPU-dirty bytes fall through to the slow drain below:
		// guest flag-poll loops (EOP/BDA-backed values) rely on that drain's
		// flush to land their writebacks; skipping it livelocked boot at
		// 43M polls during bring-up.
		if ((!is_write || IsRegionRegistered(vaddr, size)) &&
		    DownloadViaReadbackQueue(buffer, vaddr, size, is_write, current_tick)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			++diag_fast;
			diag_done("fast");
			return;
		}
	}

	// Slow path: no readback queue, possibly unsubmitted writes, or an empty
	// range — flush and drain as before.
	// Diagnostic (uncommitted): split each slow sync into queue latency (until the GPU thread runs
	// the request) and the download itself (flush + drain), per caller kind.
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	auto       diag_started  = diag_t0;
	bool       diag_dirty    = false;
	uint64_t   diag_writer   = 0;
	m_scheduler.Context().GetGpu().SendCommandSync(
	    [this, vaddr, size, is_write, on_gpu_thread, &diag_started, &diag_dirty, &diag_writer] {
		    diag_started = std::chrono::steady_clock::now();
		    if (!on_gpu_thread) {
			    // Diagnostic: the shader whose bound written buffer last covered this address.
			    auto it = g_gpu_write_ranges.upper_bound(vaddr);
			    for (int steps = 0; steps < 64 && it != g_gpu_write_ranges.begin(); steps++) {
				    --it;
				    if (it->first <= vaddr && vaddr < it->first + it->second.first) {
					    diag_writer = it->second.second;
					    break;
				    }
			    }
		    }
		    if (is_write && !IsRegionRegistered(vaddr, size)) {
			    return;
		    }
		    auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		    if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
			    diag_dirty = true;
			    m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
		    }
		    if (is_write) {
			    m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		    }
	    });
	{
		static std::mutex split_mutex;
		static double     split_queue[2]    = {}; // [guest thread, GPU thread]
		static double     split_download[2] = {};
		static uint64_t   split_calls[2]    = {};
		static uint64_t   split_dirty[2]    = {};
		static uint64_t   split_printed     = 0;
		const auto        now               = std::chrono::steady_clock::now();
		const int         kind              = on_gpu_thread ? 1 : 0;
		std::scoped_lock  lock {split_mutex};
		split_queue[kind] += std::chrono::duration<double, std::milli>(diag_started - diag_t0).count();
		split_download[kind] += std::chrono::duration<double, std::milli>(now - diag_started).count();
		split_calls[kind]++;
		split_dirty[kind] += diag_dirty ? 1 : 0;
		static std::map<std::string, std::pair<uint64_t, double>> by_reason;
		static std::map<uint64_t, std::pair<uint64_t, double>>    by_page;
		const double total_ms = std::chrono::duration<double, std::milli>(now - diag_t0).count();
		std::string reason_name = std::string(on_gpu_thread ? "gpu:" : "guest:") +
		                          (g_sync_reason != nullptr ? g_sync_reason : "other");
		if (on_gpu_thread && g_sync_cp_op >= 0 && g_sync_reason != nullptr &&
		    std::string_view(g_sync_reason).starts_with("fault")) {
			char op_text[16];
			std::snprintf(op_text, sizeof(op_text), "@op%02x", g_sync_cp_op);
			reason_name += op_text;
		}
		auto& reason_entry = by_reason[reason_name];
		reason_entry.first++;
		reason_entry.second += total_ms;
		// Diagnostic (uncommitted): cumulative totals with a wall-clock stamp every 2 s, so a
		// window (e.g. START GAME -> arena) can be cut out: [synctime] <unix ms> reason=syncs/ms ...
		static auto last_stamp = std::chrono::steady_clock::time_point {};
		if (now - last_stamp >= std::chrono::seconds(2)) {
			last_stamp      = now;
			const auto unix = std::chrono::duration_cast<std::chrono::milliseconds>(
			                      std::chrono::system_clock::now().time_since_epoch())
			                      .count();
			std::string line = "[synctime] " + std::to_string(unix);
			for (const auto& [name, entry]: by_reason) {
				line += " " + name + "=" + std::to_string(entry.first) + "/" +
				        std::to_string(static_cast<uint64_t>(entry.second));
			}
			std::fprintf(stderr, "%s\n", line.c_str());
		}
		if (on_gpu_thread) {
			auto& page_entry = by_page[vaddr >> 16u];
			page_entry.first++;
			page_entry.second += total_ms;
		}
		// Guest-thread syncs by 64 KiB page: count, ms, writes, bytes requested, last writer shader.
		struct GuestPage {
			uint64_t count = 0, writes = 0, bytes = 0, writer = 0;
			double   ms    = 0;
		};
		static std::map<uint64_t, GuestPage> by_guest_page;
		if (!on_gpu_thread) {
			auto& guest = by_guest_page[vaddr >> 16u];
			guest.count++;
			guest.writes += is_write ? 1 : 0;
			guest.bytes += size;
			guest.ms += total_ms;
			if (diag_writer != 0) guest.writer = diag_writer;
		}
		if (split_printed % 4096 == 4095) {
			std::fprintf(stderr,
			             "[syncreason] color-metadata checks covered by a known GPU fill: %llu, "
			             "not covered: %llu\n",
			             static_cast<unsigned long long>(g_meta_fill_known),
			             static_cast<unsigned long long>(g_meta_fill_unknown));
			std::fprintf(stderr,
			             "[syncreason] color-metadata covering write size: <=64K %llu, <=1M %llu, "
			             "<=16M %llu, >16M %llu, none %llu\n",
			             static_cast<unsigned long long>(g_meta_cover_hist[0]),
			             static_cast<unsigned long long>(g_meta_cover_hist[1]),
			             static_cast<unsigned long long>(g_meta_cover_hist[2]),
			             static_cast<unsigned long long>(g_meta_cover_hist[3]),
			             static_cast<unsigned long long>(g_meta_cover_hist[4]));
			std::vector<std::pair<uint64_t, uint64_t>> writers(g_meta_cover_shaders.begin(),
			                                                   g_meta_cover_shaders.end());
			std::sort(writers.begin(), writers.end(),
			          [](const auto& a, const auto& b) { return a.second > b.second; });
			for (size_t i = 0; i < std::min<size_t>(writers.size(), 6); i++) {
				std::fprintf(stderr, "[syncreason]   metadata writer cs %016llx: %llu drains\n",
				             static_cast<unsigned long long>(writers[i].first),
				             static_cast<unsigned long long>(writers[i].second));
			}
			for (const auto& [name, entry]: by_reason) {
				std::fprintf(stderr, "[syncreason] %-28s %8llu syncs %9.0f ms\n", name.c_str(),
				             static_cast<unsigned long long>(entry.first), entry.second);
			}
			std::vector<std::pair<uint64_t, std::pair<uint64_t, double>>> pages(by_page.begin(),
			                                                                     by_page.end());
			std::sort(pages.begin(), pages.end(), [](const auto& a, const auto& b) {
				return a.second.second > b.second.second;
			});
			for (size_t i = 0; i < std::min<size_t>(pages.size(), 8); i++) {
				std::fprintf(stderr, "[syncreason]   gpu-thread page 0x%012llx: %llu syncs %.0f ms\n",
				             static_cast<unsigned long long>(pages[i].first << 16u),
				             static_cast<unsigned long long>(pages[i].second.first),
				             pages[i].second.second);
			}
			std::vector<std::pair<uint64_t, GuestPage>> guest_pages(by_guest_page.begin(),
			                                                        by_guest_page.end());
			std::sort(guest_pages.begin(), guest_pages.end(),
			          [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
			for (size_t i = 0; i < std::min<size_t>(guest_pages.size(), 10); i++) {
				const auto& g = guest_pages[i].second;
				std::fprintf(stderr,
				             "[syncreason]   guest page 0x%012llx: %llu syncs (%llu writes) %.0f ms, "
				             "avg %llu bytes, writer cs %016llx\n",
				             static_cast<unsigned long long>(guest_pages[i].first << 16u),
				             static_cast<unsigned long long>(g.count),
				             static_cast<unsigned long long>(g.writes), g.ms,
				             static_cast<unsigned long long>(g.count != 0 ? g.bytes / g.count : 0),
				             static_cast<unsigned long long>(g.writer));
			}
			std::fflush(stderr);
		}
		if (++split_printed % 1024 == 0) {
			std::fprintf(stderr,
			             "[syncsplit] guest threads: %llu syncs (%llu with dirty bytes), queue %.0f ms, "
			             "download %.0f ms | GPU thread: %llu syncs (%llu dirty), download %.0f ms\n",
			             static_cast<unsigned long long>(split_calls[0]),
			             static_cast<unsigned long long>(split_dirty[0]), split_queue[0],
			             split_download[0], static_cast<unsigned long long>(split_calls[1]),
			             static_cast<unsigned long long>(split_dirty[1]), split_download[1]);
			std::fflush(stderr);
		}
	}
	diag_done("slow");
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		if (g_gpu_write_ranges.size() > 200000) {
			g_gpu_write_ranges.clear();
		}
		g_gpu_write_ranges[vaddr] = {size, g_sync_shader};
		// lower_bound: a record that starts exactly at the end of this write only touches it.
		for (auto it = g_gpu_fills.lower_bound(vaddr + size); it != g_gpu_fills.begin();) {
			--it;
			if (it->first + it->second.first <= vaddr) {
				break;
			}
			it = g_gpu_fills.erase(it);
		}
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		// Perf (uncommitted): remember the tick of the command buffer that
		// records this GPU write. Readbacks then wait on that submission's
		// timeline tick instead of draining the whole command pipeline.
		// Monotonic is sufficient: a later tick implies all earlier ones.
		m_last_write_tick.store(m_scheduler.CurrentTick(), std::memory_order_release);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		for (uint64_t offset = vaddr; offset < vaddr + size && offset < 32; offset += 4) {
			g_gds_writes[offset / 4] = {1, value, 0, ++g_gds_write_serial};
		}
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, 1);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	g_gpu_fills[vaddr] = {size, value};
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, 2);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	if (dst_gds && src_memory) {
		const bool gpu_written = IsRegionGpuModified(src_vaddr, size);
		for (uint64_t offset = dst_vaddr; offset < dst_vaddr + size && offset < 32; offset += 4) {
			const auto source = src_vaddr + (offset - dst_vaddr);
			uint32_t   value  = 0xffffffffu;
			if (!gpu_written) {
				(void)Libs::LibKernel::Memory::TryReadBacking(source, &value, sizeof(value));
			}
			g_gds_writes[offset / 4] = {gpu_written ? 3 : 2, value, source, ++g_gds_write_serial};
		}
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

void BufferCache::ReadGdsDwords(uint32_t* out, uint32_t count) {
	// Stream memory: read the mapping directly (Invalidate() is for download buffers).
	std::memcpy(out, m_gds_buffer.Mapped().data(), uint64_t {count} * sizeof(uint32_t));
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::LogMemoryStats() const {
	size_t   count   = 0;
	uint64_t total   = 0;
	uint64_t largest = 0;
	m_slot_buffers.ForEach([&](BufferId, const Buffer& buffer) {
		if (buffer.is_deleted) {
			return;
		}
		count++;
		total += buffer.Size();
		largest = std::max(largest, buffer.Size());
	});
	std::fprintf(stderr,
	             "[imgdiag] buffer cache: %zu buffers, %" PRIu64 " MiB, largest %" PRIu64
	             " MiB (+ BDA page table %" PRIu64 " MiB)\n",
	             count, total >> 20u, largest >> 20u, m_bda_pagetable_buffer.Size() >> 20u);
	std::fflush(stderr);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

/// Initialize the fault bitset before scheduling fault collection.
void BufferCache::ProcessFaultBuffer() {
	EnsureBdaStorageInitialized();
	m_fault_manager.ProcessFaultBuffer();
}

/// Upload pending guest writes only after the BDA storage has been initialized.
void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	EnsureBdaStorageInitialized();
	// Take before walking: writes queued during synchronization remain pending.
	const auto pending = m_memory_tracker.TakeBdaSync();
	pending.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			SynchronizeBuffersInRange(start, finish - start);
		});
	});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
