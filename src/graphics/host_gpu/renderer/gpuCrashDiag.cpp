#include "graphics/host_gpu/renderer/gpuCrashDiag.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace Libs::Graphics::GpuCrashDiag {

namespace {

constexpr uint64_t RingSize = 1u << 16u;

struct State {
	int                                    level              = -1;
	bool                                   checkpoints        = false;
	bool                                   device_fault       = false;
	bool                                   fault_vendor_binary = false;
	bool                                   diagnostics_config = false;
	GraphicContext*                        graphics           = nullptr;
	std::atomic<uint64_t>                  next_seq {1};
	std::unique_ptr<std::array<Record, RingSize>> ring;
	std::atomic_flag                       reported = ATOMIC_FLAG_INIT;

	vk::PhysicalDeviceFaultFeaturesEXT       fault_features {};
	vk::DeviceDiagnosticsConfigCreateInfoNV  diagnostics {};
};

State& GetState() {
	static State state;
	return state;
}

int Level() {
	auto& s = GetState();
	if (s.level < 0) {
		const char* env = std::getenv("KYTY_GPU_CRASH_DIAG");
		s.level         = env != nullptr ? std::atoi(env) : 0;
	}
	return s.level;
}

const char* OpName(uint32_t op) {
	switch (op) {
		case 0: return "DispatchDirect";
		case 1: return "DrawIndex";
		case 2: return "DrawIndexAuto";
		case 9: return "DispatchIndirect";
		default: return "Op?";
	}
}

const char* AddressTypeName(VkDeviceFaultAddressTypeEXT type) {
	switch (type) {
		case VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT: return "NONE";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT: return "READ_INVALID";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT: return "WRITE_INVALID";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT: return "EXECUTE_INVALID";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_UNKNOWN_EXT: return "IP_UNKNOWN";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_INVALID_EXT: return "IP_INVALID";
		case VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT: return "IP_FAULT";
		default: return "?";
	}
}

class Writer {
public:
	Writer() { m_file = std::fopen("_gpu_crash_report.txt", "w"); }
	~Writer() {
		if (m_file != nullptr) {
			std::fclose(m_file);
		}
	}
	Writer(const Writer&)            = delete;
	Writer& operator=(const Writer&) = delete;

	template <typename... Args>
	void operator()(const char* format, Args... args) {
		const auto text = fmt::sprintf(format, args...);
		LOGF("%s", text.c_str());
		std::fputs(text.c_str(), stdout);
		if (m_file != nullptr) {
			std::fputs(text.c_str(), m_file);
			std::fflush(m_file);
		}
	}

private:
	std::FILE* m_file = nullptr;
};

void PrintRecord(Writer& out, uint64_t seq, const char* tag) {
	auto& s = GetState();
	if (!s.ring || seq == 0) {
		return;
	}
	const auto& r = (*s.ring)[seq & (RingSize - 1u)];
	out("  %s#%" PRIu64 " %s submit=%" PRIu64 " frame=%" PRIu64 " tick=%" PRIu64
	    " hash=0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 " args=%u,%u,%u,0x%x addr=0x%016" PRIx64
	    "\n",
	    tag, seq, OpName(r.op), r.submit_id, r.frame, r.tick, r.hash[0], r.hash[1], r.hash[2],
	    r.args[0], r.args[1], r.args[2], r.args[3], r.addr);
}

void ReportCheckpoints(Writer& out, GraphicContext& g) {
	auto& s = GetState();
	if (!s.checkpoints || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV == nullptr) {
		out("checkpoints: unavailable\n");
		return;
	}
	const auto queue = static_cast<VkQueue>(g.queue);
	uint32_t   count = 0;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, nullptr);
	std::vector<VkCheckpointDataNV> data(count, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
	if (count != 0) {
		VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueueCheckpointDataNV(queue, &count, data.data());
	}
	const auto issued = s.next_seq.load() - 1u;
	out("checkpoints: %u entries, last issued marker=#%" PRIu64 "\n", count, issued);
	uint64_t lo = UINT64_MAX;
	uint64_t hi = 0;
	for (const auto& d: data) {
		const auto seq = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d.pCheckpointMarker));
		out(" stage=%s marker=#%" PRIu64 "\n",
		    vk::to_string(static_cast<vk::PipelineStageFlagBits>(d.stage)).c_str(), seq);
		PrintRecord(out, seq, "");
		if (seq != 0) {
			lo = std::min(lo, seq);
			hi = std::max(hi, seq);
		}
	}
	if (hi == 0) {
		return;
	}
	// Everything between the oldest and newest reported marker (plus a little lookahead) was
	// potentially in flight when the device died.
	const uint64_t first = lo > 8 ? lo - 8 : 1;
	const uint64_t last  = std::min<uint64_t>(issued, hi + 8);
	out("in-flight window #%" PRIu64 "..#%" PRIu64 ":\n", first, last);
	for (uint64_t seq = first; seq <= last && last - first < RingSize; seq++) {
		PrintRecord(out, seq, seq >= lo && seq <= hi ? "*" : " ");
	}
}

void ReportFault(Writer& out, GraphicContext& g) {
	auto& s = GetState();
	if (!s.device_fault || VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		out("device fault: unavailable\n");
		return;
	}
	const auto device = static_cast<VkDevice>(g.device);
	VkDeviceFaultCountsEXT counts {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
	auto result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, nullptr);
	out("device fault: counts result=%d addresses=%u vendor=%u binary=%" PRIu64 "\n",
	    static_cast<int>(result), counts.addressInfoCount, counts.vendorInfoCount,
	    static_cast<uint64_t>(counts.vendorBinarySize));
	if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
		return;
	}
	if (!s.fault_vendor_binary) {
		counts.vendorBinarySize = 0;
	}
	std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
	std::vector<uint8_t>                     binary(counts.vendorBinarySize);
	VkDeviceFaultInfoEXT                     info {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
	info.pAddressInfos     = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos      = vendors.empty() ? nullptr : vendors.data();
	info.pVendorBinaryData = binary.empty() ? nullptr : binary.data();
	result = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT(device, &counts, &info);
	out("device fault: info result=%d description=\"%s\"\n", static_cast<int>(result),
	    info.description);
	for (uint32_t i = 0; i < counts.addressInfoCount; i++) {
		const auto& a = addresses[i];
		out("  address[%u]: type=%s addr=0x%016" PRIx64 " precision=0x%" PRIx64 "\n", i,
		    AddressTypeName(a.addressType), static_cast<uint64_t>(a.reportedAddress),
		    static_cast<uint64_t>(a.addressPrecision));
	}
	for (uint32_t i = 0; i < counts.vendorInfoCount; i++) {
		const auto& v = vendors[i];
		out("  vendor[%u]: \"%s\" code=0x%016" PRIx64 " data=0x%016" PRIx64 "\n", i, v.description,
		    static_cast<uint64_t>(v.vendorFaultCode), static_cast<uint64_t>(v.vendorFaultData));
	}
	if (!binary.empty()) {
		if (auto* file = std::fopen("_gpu_crash_vendor.bin", "wb"); file != nullptr) {
			std::fwrite(binary.data(), 1, binary.size(), file);
			std::fclose(file);
			out("  vendor binary: %zu bytes -> _gpu_crash_vendor.bin\n", binary.size());
		}
	}
}

} // namespace

bool Enabled() {
	return Level() > 0;
}

void SelectDeviceExtensions(const std::vector<vk::ExtensionProperties>& available,
                            std::vector<const char*>&                   extensions) {
	if (!Enabled()) {
		return;
	}
	auto&      s   = GetState();
	const auto has = [&](const char* name) {
		return std::ranges::any_of(available, [&](const vk::ExtensionProperties& e) {
			return std::strcmp(e.extensionName.data(), name) == 0;
		});
	};
	if (has(VK_EXT_DEVICE_FAULT_EXTENSION_NAME)) {
		extensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
		s.device_fault = true;
	}
	if (has(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME)) {
		extensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
		s.checkpoints = true;
		s.ring        = std::make_unique<std::array<Record, RingSize>>();
	}
	if (Level() >= 2 && has(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME)) {
		extensions.push_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
		s.diagnostics_config = true;
	}
	LOGF("GpuCrashDiag: level=%d device_fault=%s checkpoints=%s diagnostics_config=%s\n", Level(),
	     s.device_fault ? "on" : "off", s.checkpoints ? "on" : "off",
	     s.diagnostics_config ? "on" : "off");
}

void ChainDeviceCreateInfo(vk::DeviceCreateInfo& create_info) {
	auto& s = GetState();
	if (s.device_fault) {
		s.fault_features.deviceFault             = VK_TRUE;
		s.fault_features.deviceFaultVendorBinary = VK_TRUE;
		s.fault_features.pNext                   = const_cast<void*>(create_info.pNext);
		create_info.pNext                        = &s.fault_features;
		s.fault_vendor_binary                    = true;
	}
	if (s.diagnostics_config) {
		s.diagnostics.flags = vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderDebugInfo |
		                      vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableResourceTracking |
		                      vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableAutomaticCheckpoints |
		                      vk::DeviceDiagnosticsConfigFlagBitsNV::eEnableShaderErrorReporting;
		s.diagnostics.pNext = const_cast<void*>(create_info.pNext);
		create_info.pNext   = &s.diagnostics;
	}
}

void Install(GraphicContext* graphics) {
	GetState().graphics = graphics;
}

void Mark(vk::CommandBuffer buffer, const Record& record) {
	auto& s = GetState();
	if (!s.checkpoints || VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV == nullptr) {
		return;
	}
	const auto seq                    = s.next_seq.fetch_add(1, std::memory_order_relaxed);
	(*s.ring)[seq & (RingSize - 1u)] = record;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdSetCheckpointNV(
	    static_cast<VkCommandBuffer>(buffer),
	    reinterpret_cast<const void*>(static_cast<uintptr_t>(seq)));
}

void Report(const char* where, vk::Result result) {
	auto& s = GetState();
	if (!Enabled() || s.graphics == nullptr || s.reported.test_and_set()) {
		return;
	}
	Writer out;
	out("=== GpuCrashDiag report: %s -> %s (%d) ===\n", where, vk::to_string(result).c_str(),
	    static_cast<int>(result));
	ReportFault(out, *s.graphics);
	ReportCheckpoints(out, *s.graphics);
	out("=== end GpuCrashDiag report ===\n");
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuCrashDiag
