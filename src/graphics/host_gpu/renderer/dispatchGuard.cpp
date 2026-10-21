#include "graphics/host_gpu/renderer/dispatchGuard.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/gpu_dispatch_args_guard_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace Libs::Graphics {

namespace {

// 16-byte slots; a slot is reused only after this many guarded dispatches.
constexpr uint64_t SlotRingSize = 64ull * 1024;

struct GuardParameters {
	uint32_t arguments_base = 0;
	uint32_t guarded_base   = 0;
	uint32_t max_x          = 0;
	uint32_t max_y          = 0;
	uint32_t max_z          = 0;
	uint32_t max_total      = 0;
};

} // namespace

DispatchArgsGuard::DispatchArgsGuard(GraphicContext& graphics, CommandScheduler& scheduler,
                                     uint32_t max_total)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_slots(graphics, scheduler, MemoryUsage::DeviceLocal, 0,
              vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer,
              SlotRingSize),
      m_stats(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eStorageBuffer,
              sizeof(uint32_t)),
      m_max_total(max_total) {
	SetVulkanObjectNameF(m_graphics.device, m_slots.Handle(), "Kyty.DispatchGuardSlots");
	SetVulkanObjectNameF(m_graphics.device, m_stats.Handle(), "Kyty.DispatchGuardStats");
	// The only host write, before any GPU access.
	const uint32_t zero = 0;
	std::memcpy(m_stats.Mapped().data(), &zero, sizeof(zero));
	m_stats.Flush(0, sizeof(zero));
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t index = 0; index < bindings.size(); index++) {
		bindings[index] = {index, vk::DescriptorType::eStorageBuffer, 1,
		                   vk::ShaderStageFlagBits::eCompute, nullptr};
	}
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_set_layout),
	    "create dispatch guard descriptor layout");
	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                         sizeof(GuardParameters)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	    "create dispatch guard pipeline layout");
	const auto module = CompileSPV(GPU_DISPATCH_ARGS_GUARD_SPV, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create dispatch guard pipeline");
}

DispatchArgsGuard::~DispatchArgsGuard() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

DispatchArgsGuard::Slot DispatchArgsGuard::Record(vk::CommandBuffer command,
                                                  const Buffer&     arguments,
                                                  uint64_t          arguments_offset) {
	EXIT_IF(arguments_offset % 4 != 0);
	const auto storage_alignment = std::max<uint64_t>(
	    m_graphics.GetPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment, 16);
	auto start = Common::AlignUp(m_cursor, storage_alignment);
	if (start + 16 > m_slots.Size()) {
		start = 0;
	}
	m_cursor = start + 16;

	// Guest addresses are only dword aligned: bind from an aligned offset, index past it.
	const auto arguments_descriptor = Common::AlignDown(arguments_offset, storage_alignment);
	const auto arguments_base       = arguments_offset - arguments_descriptor;
	const auto& limits              = m_graphics.GetPhysicalDeviceProperties().limits;
	GuardParameters parameters {.arguments_base = static_cast<uint32_t>(arguments_base / 4),
	                            .guarded_base   = 0,
	                            .max_x          = limits.maxComputeWorkGroupCount[0],
	                            .max_y          = limits.maxComputeWorkGroupCount[1],
	                            .max_z          = limits.maxComputeWorkGroupCount[2],
	                            .max_total      = m_max_total};

	// Argument writes become visible; earlier dispatches stop reading the reused slot.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {arguments.Handle(), arguments_descriptor, arguments_base + 3 * sizeof(uint32_t)},
	    {m_slots.Handle(), start, 16},
	    {m_stats.Handle(), 0, sizeof(uint32_t)},
	};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                      sizeof(parameters), &parameters);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask         = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask        = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask         = vk::PipelineStageFlagBits2::eDrawIndirect |
	                             vk::PipelineStageFlagBits2::eAllCommands |
	                             vk::PipelineStageFlagBits2::eHost;
	after.dstAccessMask        = vk::AccessFlagBits2::eIndirectCommandRead |
	                             vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eHostRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);
	return {.buffer = m_slots.Handle(), .offset = start};
}

uint32_t DispatchArgsGuard::Zeroed() const {
	uint32_t zeroed = 0;
	std::memcpy(&zeroed, m_stats.Mapped().data(), sizeof(zeroed));
	return zeroed;
}

} // namespace Libs::Graphics
