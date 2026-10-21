#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/frameTrace.h"
#include "graphics/host_gpu/renderer/gpuCrashDiag.h"

#include <chrono>
#include <cstdio>
#include <thread>

namespace Libs::Graphics {

// Diagnostic (uncommitted, [gdsdiag]): the hair work queue (cs ea94f09b2981eb43) keeps its item
// limit in GDS[0] and its counter in GDS[4]. At a hang, read them twice 0.5 s apart from the
// host-visible GDS buffer: a huge limit, a counter racing past the limit, or a stuck counter.
static void ProbeGdsAtHang(const char* where) {
	if (g_diag_buffer_cache == nullptr) {
		return;
	}
	for (int sample = 0; sample < 2; sample++) {
		uint32_t gds[8] = {};
		g_diag_buffer_cache->ReadGdsDwords(gds, 8);
		std::fprintf(stderr,
		             "[gdsdiag] at %s (sample %d, hair dispatches %llu): %08x %08x %08x %08x "
		             "%08x %08x %08x %08x\n",
		             where, sample, static_cast<unsigned long long>(g_gds_hair_dispatches), gds[0],
		             gds[1], gds[2], gds[3], gds[4], gds[5], gds[6], gds[7]);
		std::fflush(stderr);
		if (sample == 0) {
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}
	}
	const auto& w = g_gds_writes[0];
	std::fprintf(stderr, "[gdsdiag] last CP write to GDS[0]: kind %d value %08x source %llx\n",
	             w.kind, w.value, static_cast<unsigned long long>(w.source));
	// The recent hair dispatches, oldest first (renderCompute.cpp records them).
	const auto last = g_gds_hair_dispatches;
	for (uint64_t n = last > 63 ? last - 63 : 1; n <= last; n++) {
		const auto& r = g_gds_hair_ring[n % 64];
		if (r.n != n) {
			continue;
		}
		std::fprintf(stderr,
		             "[gdsring] hair #%llu frame %d | CP write to GDS[0] before: #%llu kind %d src "
		             "%llx | after: limit %u counter %u%s\n",
		             static_cast<unsigned long long>(r.n), r.frame,
		             static_cast<unsigned long long>(r.serial), r.kind,
		             static_cast<unsigned long long>(r.source), r.limit, r.counter,
		             r.done ? "" : " (not completed)");
	}
	std::fflush(stderr);
}

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		GpuCrashDiag::Report("MasterSemaphore::Refresh", result);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const bool trace = GuestGpu::IsGpuThread();
	if (trace) {
		FrameTrace('W', tick);
	}
	// Diagnostic (uncommitted): NHL 27 sometimes hangs the GPU without a device loss (a shader
	// loops forever; instructions 8.7), so the crash report never ran. After 10 s of waiting, write
	// it once (NV checkpoints name the in-flight draws/dispatches) and keep waiting.
	vk::Result result = vk::Result::eTimeout;
	for (;;) {
		result = m_graphics.device.waitSemaphores(&wait_info, 10'000'000'000ull);
		if (result != vk::Result::eTimeout) {
			break;
		}
		std::fprintf(stderr, "[hangwatch] GPU wait for tick %llu exceeded 10 s\n",
		             static_cast<unsigned long long>(tick));
		std::fflush(stderr);
		ProbeGdsAtHang("hangwatch");
		GpuCrashDiag::Report("MasterSemaphore::Wait hang watchdog (>10 s)", vk::Result::eTimeout);
	}
	if (result != vk::Result::eSuccess) {
		ProbeGdsAtHang("device lost");
		GpuCrashDiag::Report("MasterSemaphore::Wait", result);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (trace) {
		FrameTrace('w', tick);
	}
	Refresh();
}

} // namespace Libs::Graphics
