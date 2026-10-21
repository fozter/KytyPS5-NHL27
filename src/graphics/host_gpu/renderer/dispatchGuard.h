#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHGUARD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHGUARD_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <cstdint>

namespace Libs::Graphics {

// Experiment (uncommitted): copies guest indirect dispatch arguments into a host-owned slot on
// the GPU, zeroed when a group count exceeds the device limits or the total group budget, so a
// dispatch whose arguments hold leftover data neither runs for seconds nor hits undefined
// behaviour (NHL 27; pengui07/KytyPS5-nhl guards the same dispatches).
class DispatchArgsGuard {
public:
	struct Slot {
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
	};

	DispatchArgsGuard(GraphicContext& graphics, CommandScheduler& scheduler, uint32_t max_total);
	~DispatchArgsGuard();
	KYTY_CLASS_NO_COPY(DispatchArgsGuard);

	// Records the guard pass; the dispatch then reads the returned slot.
	[[nodiscard]] Slot Record(vk::CommandBuffer command, const Buffer& arguments,
	                          uint64_t arguments_offset);
	// Dispatches zeroed so far, as last written by the GPU (no synchronization; diagnostics).
	[[nodiscard]] uint32_t Zeroed() const;

private:
	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	Buffer                  m_slots;
	Buffer                  m_stats;
	uint64_t                m_cursor          = 0;
	uint32_t                m_max_total       = 0;
	vk::DescriptorSetLayout m_set_layout      = nullptr;
	vk::PipelineLayout      m_pipeline_layout = nullptr;
	vk::Pipeline            m_pipeline        = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHGUARD_H_
