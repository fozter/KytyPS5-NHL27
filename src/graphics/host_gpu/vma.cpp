#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	// Experiment (uncommitted), idea from chenxiao07/KytyPS5 4acd26af: on a GPU with less than
	// 13 GiB of video memory, 64 MiB VMA blocks instead of 256 MiB. Past its budget the driver keeps
	// whole blocks in system memory; smaller blocks keep more of each frame's data in video memory
	// (NHL 27 spills 2-3 GB on an 8 GB card). KYTY_VMA_BLOCK_MB=<n> sets n MiB (256 = VMA default).
	{
		const auto     memory = physical_device.getMemoryProperties();
		vk::DeviceSize video  = 0;
		for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
			if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				video = std::max(video, memory.memoryHeaps[heap].size);
			}
		}
		const char* text = std::getenv("KYTY_VMA_BLOCK_MB");
		if (text != nullptr && std::strtoull(text, nullptr, 10) > 0) {
			info.preferredLargeHeapBlockSize = std::strtoull(text, nullptr, 10) << 20u;
		} else if (video != 0 && video < (vk::DeviceSize {13} << 30u)) {
			info.preferredLargeHeapBlockSize = vk::DeviceSize {64} << 20u;
		}
		std::printf("[vmablock] largest video heap %llu MiB, VMA block size %llu MiB\n",
		            static_cast<unsigned long long>(video >> 20u),
		            static_cast<unsigned long long>(
		                (info.preferredLargeHeapBlockSize != 0 ? info.preferredLargeHeapBlockSize
		                                                       : vk::DeviceSize {256} << 20u) >>
		                20u));
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
		// Diagnostic (uncommitted): the same line on stderr, which survives Silent printf.
		std::fprintf(stderr,
		             "[imgdiag] VMA heap %u: usage=%" PRIu64 " MiB budget=%" PRIu64
		             " MiB allocations=%" PRIu64 " MiB (%u) blocks=%" PRIu64 " MiB (%u)\n",
		             i, static_cast<uint64_t>(budgets[i].usage) >> 20u,
		             static_cast<uint64_t>(budgets[i].budget) >> 20u,
		             static_cast<uint64_t>(budgets[i].statistics.allocationBytes) >> 20u,
		             budgets[i].statistics.allocationCount,
		             static_cast<uint64_t>(budgets[i].statistics.blockBytes) >> 20u,
		             budgets[i].statistics.blockCount);
	}
	std::fflush(stderr);
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
                                 bool allow_host_memory) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	// VMA tries the other compatible memory types when the preferred one is exhausted.
	if (allow_host_memory) {
		alloc_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	} else {
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	}
	// Experiment (uncommitted): large images get their own VkDeviceMemory. Inside shared blocks,
	// evicting them only leaves holes, and a new large image may fit none of them (NHL 27 replay
	// on an 8 GB GPU: 1.4 GB free inside blocks, a 63 MiB render target still out of memory).
	constexpr vk::DeviceSize                DEDICATED_SIZE = 16ull << 20u;
	vk::DeviceImageMemoryRequirements       requirements_query {};
	requirements_query.pCreateInfo = &image_info;
	if (device.getImageMemoryRequirements(requirements_query).memoryRequirements.size >=
	    DEDICATED_SIZE) {
		alloc_info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
	}

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const auto        result       = static_cast<vk::Result>(
	    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
	                   &alloc_info, &native_image, &image.allocation, nullptr));
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		// Diagnostic (uncommitted): the failure reason must survive Silent printf
		// direction, so this goes to stderr directly rather than LOGF.
		std::fprintf(stderr,
		             "[imgdiag] vmaCreateImage failed: %s (extent=%ux%ux%u fmt=%d layers=%u "
		             "levels=%u usage=0x%x flags=0x%x type=%d)\n",
		             vk::to_string(result).c_str(), image_info.extent.width,
		             image_info.extent.height, image_info.extent.depth,
		             static_cast<int>(image_info.format), image_info.arrayLayers,
		             image_info.mipLevels,
		             static_cast<vk::ImageUsageFlags::MaskType>(image_info.usage),
		             static_cast<vk::ImageCreateFlags::MaskType>(image_info.flags),
		             static_cast<int>(image_info.imageType));
		std::fflush(stderr);
		LogMemoryBudget();
		return false;
	}

	// Diagnostic (uncommitted): each new large image size once, to see the game's render size.
	if (image_info.extent.width >= 1024 && image_info.extent.height >= 512) {
		static std::mutex              sizes_mutex;
		static std::set<uint64_t>      sizes;
		const uint64_t                 key = (uint64_t {image_info.extent.width} << 32u) |
		                                     image_info.extent.height;
		const std::scoped_lock         lock {sizes_mutex};
		if (sizes.insert(key).second) {
			std::fprintf(stderr, "[ressizediag] new image size %ux%u fmt=%d\n",
			             image_info.extent.width, image_info.extent.height,
			             static_cast<int>(image_info.format));
			std::fflush(stderr);
		}
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
