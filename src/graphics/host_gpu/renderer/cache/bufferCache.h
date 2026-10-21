#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Diagnostic (uncommitted): who requested the current BufferCache::ReadMemory on this thread.
inline thread_local const char* g_sync_reason = nullptr;
// PM4 opcode the GPU thread is executing (-1 outside packet handlers).
inline thread_local int g_sync_cp_op = -1;
// GPU-side fills still intact: address -> (size, value). GPU thread only.
inline std::map<uint64_t, std::pair<uint64_t, uint32_t>> g_gpu_fills;
inline uint64_t g_meta_fill_known = 0, g_meta_fill_unknown = 0;
// Diagnostic (uncommitted, [gdsdiag]): the last command-processor write to each of the first 8
// GDS dwords: kind 0 none, 1 fill (value), 2 copy from memory (source address; value when the CPU
// could read it, else 0xffffffff), 3 copy from a GPU-written source. GPU thread only.
struct GdsDiagWrite {
	int      kind   = 0;
	uint32_t value  = 0;
	uint64_t source = 0;
	uint64_t serial = 0;
};
inline GdsDiagWrite g_gds_writes[8] = {};
inline uint64_t     g_gds_write_serial = 0;
inline uint64_t     g_gds_hair_dispatches = 0;
// The last 64 hair dispatches: the last CP write to GDS[0] before each (its serial tells whether a
// reset came in between) and GDS[0] (limit) / GDS[4] (counter) after it completed. Printed by the
// hang probe: a limit growing over dispatches without a new CP write = a lost reset.
struct GdsHairRecord {
	uint64_t n       = 0;
	int      frame   = 0;
	int      kind    = 0;
	uint64_t source  = 0;
	uint64_t serial  = 0;
	uint32_t limit   = 0;
	uint32_t counter = 0;
	bool     done    = false;
};
inline GdsHairRecord g_gds_hair_ring[64] = {};
class BufferCache;
// The live buffer cache, for the hang probe in MasterSemaphore::Wait (diagnostic).
inline BufferCache* g_diag_buffer_cache = nullptr;
// Compute shader whose bindings are being acquired (0 outside compute dispatch preparation).
inline thread_local uint64_t g_sync_shader = 0;
struct SyncShaderScope {
	explicit SyncShaderScope(uint64_t hash) { g_sync_shader = hash; }
	~SyncShaderScope() { g_sync_shader = 0; }
	SyncShaderScope(const SyncShaderScope&)            = delete;
	SyncShaderScope& operator=(const SyncShaderScope&) = delete;
};
// Shader whose resources are being specialized, and the specialization reads that needed a GPU
// readback so far (diagnostic, GPU thread).
inline thread_local uint64_t g_srt_reader    = 0;
inline thread_local uint64_t g_srt_readbacks = 0;
// Recent GPU write markings: start -> (size, shader). GPU thread only.
inline std::map<uint64_t, std::pair<uint64_t, uint64_t>> g_gpu_write_ranges;
// Color-metadata drains by the size of the write marking that covers the metadata:
// <=64 KiB, <=1 MiB, <=16 MiB, >16 MiB, none found.
inline uint64_t g_meta_cover_hist[5] = {};
inline std::map<uint64_t, uint64_t> g_meta_cover_shaders;
struct SyncReasonScope {
	const char* previous;
	explicit SyncReasonScope(const char* reason): previous(g_sync_reason) { g_sync_reason = reason; }
	~SyncReasonScope() { g_sync_reason = previous; }
	SyncReasonScope(const SyncReasonScope&)            = delete;
	SyncReasonScope& operator=(const SyncReasonScope&) = delete;
};

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	/// Return initialized BDA mappings; requires an active recording command buffer.
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept;
	/// Return initialized page-fault bits; requires an active recording command buffer.
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept;
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	void               SynchronizeBda(const RangeSet& mapped);
	void               QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	void               RunGarbageCollector();
	// Diagnostic (uncommitted): the first dwords of the GDS buffer as the CPU sees them now.
	void               ReadGdsDwords(uint32_t* out, uint32_t count);
	// Diagnostic (uncommitted): buffer count, total and largest size on stderr.
	void               LogMemoryStats() const;

private:
	friend struct BufferCacheTestAccess;
	/// Schedule the one-time clear before BDA mappings or faults can be consumed.
	void EnsureBdaStorageInitialized();

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	// Perf (uncommitted): readback-queue download; see ReadMemory.
	bool DownloadViaReadbackQueue(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_write,
	                              uint64_t wait_tick);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	bool                                              m_bda_storage_initialized = false;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	// Tick of the command buffer that recorded the latest GPU write into
	// m_gpu_modified_ranges (monotonic). Readbacks wait on it instead of
	// draining the whole pipeline.
	std::atomic<uint64_t>                             m_last_write_tick {0};
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
