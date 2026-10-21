#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMETRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMETRACE_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

// Diagnostic (uncommitted): KYTY_FRAME_TRACE=<file> records frame pacing events with microsecond
// timestamps: submissions enqueued by the game (G graphics, C compute, F flip preparation,
// S suspend point), the GPU thread beginning/ending each one (b/e, x/y for guest-thread commands),
// the game waiting at a suspend point (P) and continuing (A), the GPU completing a suspend point
// (R), EOP interrupts queued (q) and fired (I), and waits for the GPU (W/w, the tick waited for).
// Line: "<us> <kind> <arg>"; a header line maps the clock to unix milliseconds.
inline void FrameTrace(char kind, uint64_t arg) {
	static FILE* out = [] {
		const char* path = std::getenv("KYTY_FRAME_TRACE");
		FILE*       file = path != nullptr ? std::fopen(path, "w") : nullptr;
		if (file != nullptr) {
			const auto steady = std::chrono::duration_cast<std::chrono::microseconds>(
			                        std::chrono::steady_clock::now().time_since_epoch())
			                        .count();
			const auto unix_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			                         std::chrono::system_clock::now().time_since_epoch())
			                         .count();
			std::fprintf(file, "# frame trace: steady us %lld = unix ms %lld\n",
			             static_cast<long long>(steady), static_cast<long long>(unix_ms));
		}
		return file;
	}();
	if (out == nullptr) {
		return;
	}
	static std::atomic<long long> flushed {0};
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	std::fprintf(out, "%lld %c %llu\n", static_cast<long long>(us), kind,
	             static_cast<unsigned long long>(arg));
	if (us - flushed.load(std::memory_order_relaxed) > 1'000'000) {
		flushed.store(us, std::memory_order_relaxed);
		std::fflush(out);
	}
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FRAMETRACE_H_ */
