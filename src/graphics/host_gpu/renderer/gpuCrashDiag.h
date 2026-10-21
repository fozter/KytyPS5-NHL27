#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCRASHDIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCRASHDIAG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// Opt-in GPU crash diagnostics (KYTY_GPU_CRASH_DIAG=1): NV checkpoints before every draw and
// dispatch plus VK_EXT_device_fault, reported when the device is lost.
namespace GpuCrashDiag {

struct Record {
	uint32_t op        = 0;
	uint64_t submit_id = 0;
	uint64_t frame     = 0;
	uint64_t tick      = 0;
	uint64_t hash[3]   = {}; // CS, or VS/PS/extra stage
	uint32_t args[4]   = {}; // groups x/y/z, mode | index/vertex count, instances, ...
	uint64_t addr      = 0;  // CS data address or index address
};

[[nodiscard]] bool Enabled();

// Device creation hooks.
void SelectDeviceExtensions(const std::vector<vk::ExtensionProperties>& available,
                            std::vector<const char*>&                   extensions);
void ChainDeviceCreateInfo(vk::DeviceCreateInfo& create_info);
void Install(GraphicContext* graphics);

void Mark(vk::CommandBuffer buffer, const Record& record);

// Dumps checkpoints and fault info once; safe to call from any device-lost path.
void Report(const char* where, vk::Result result);

} // namespace GpuCrashDiag

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUCRASHDIAG_H_
