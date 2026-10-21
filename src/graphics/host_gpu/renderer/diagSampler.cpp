#include "graphics/host_gpu/renderer/diagSampler.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#undef min
#undef max
#endif

namespace Libs::Graphics::DiagSampler {

#if defined(_WIN32)
namespace {

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

constexpr int    MaxFrames    = 48;
constexpr size_t FlushSamples = 4096;

struct Target {
	HANDLE   thread     = nullptr;
	uint64_t stack_low  = 0;
	uint64_t stack_high = 0;
};

struct Module {
	uint64_t                base      = 0;
	uint64_t                end       = 0;
	const RUNTIME_FUNCTION* functions = nullptr;
	uint32_t                count     = 0;
};

std::atomic<const Target*> g_target {nullptr};

// 1: threads inside a Scope, every 0.5 ms. 2: the first thread that enters a Scope (the GPU
// thread), for the rest of the run, every 2 ms. 3: every thread of the process, every 10 ms.
int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_DIAG_SAMPLER");
		return value == nullptr ? 0 : value[0] >= '1' && value[0] <= '3' ? value[0] - '0' : 0;
	}();
	return mode;
}

uint64_t UnixMilliseconds() {
	FILETIME time {};
	GetSystemTimePreciseAsFileTime(&time);
	const auto ticks = (static_cast<uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
	return (ticks - 116444736000000000ull) / 10000;
}

// Never freed: the sampler may still hold the pointer after the scope ends.
const Target* CurrentTarget() {
	thread_local const Target* target = [] {
		auto* created = new Target;
		DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &created->thread,
		                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0);
		using Limits = VOID(WINAPI*)(PULONG_PTR, PULONG_PTR);
		const auto limits = reinterpret_cast<Limits>(reinterpret_cast<void*>(
		    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetCurrentThreadStackLimits")));
		ULONG_PTR low = 0, high = 0;
		if (limits != nullptr) {
			limits(&low, &high);
		} else {
			const auto* tib = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
			low             = reinterpret_cast<ULONG_PTR>(tib->StackLimit);
			high            = reinterpret_cast<ULONG_PTR>(tib->StackBase);
		}
		created->stack_low  = low;
		created->stack_high = high;
		return created;
	}();
	return target;
}

std::vector<Module> ListModules(FILE* out) {
	std::vector<HMODULE> handles(1024);
	DWORD                needed = 0;
	if (!K32EnumProcessModules(GetCurrentProcess(), handles.data(),
	                           static_cast<DWORD>(handles.size() * sizeof(HMODULE)), &needed)) {
		return {};
	}
	handles.resize(std::min<size_t>(handles.size(), needed / sizeof(HMODULE)));
	std::vector<Module> modules;
	for (auto* handle: handles) {
		MODULEINFO info {};
		if (!K32GetModuleInformation(GetCurrentProcess(), handle, &info, sizeof(info))) continue;
		Module module;
		module.base     = reinterpret_cast<uint64_t>(info.lpBaseOfDll);
		module.end      = module.base + info.SizeOfImage;
		const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(module.base);
		const auto* nt  = reinterpret_cast<const IMAGE_NT_HEADERS64*>(module.base + dos->e_lfanew);
		const auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION];
		module.functions = reinterpret_cast<const RUNTIME_FUNCTION*>(module.base + dir.VirtualAddress);
		module.count     = dir.VirtualAddress != 0 ? dir.Size / sizeof(RUNTIME_FUNCTION) : 0;
		modules.push_back(module);
		char name[MAX_PATH] {};
		GetModuleFileNameA(handle, name, MAX_PATH);
		// The PE timestamp lets the symbolizer refuse a rebuilt binary.
		std::fprintf(out, "m %" PRIx64 " %" PRIx64 " t%08x %s\n", module.base, module.end - module.base,
		             static_cast<unsigned>(nt->FileHeader.TimeDateStamp), name);
	}
	std::sort(modules.begin(), modules.end(),
	          [](const Module& a, const Module& b) { return a.base < b.base; });
	std::fflush(out);
	return modules;
}

const Module* FindModule(const Module* modules, size_t count, uint64_t pc) {
	size_t low = 0, high = count;
	while (low < high) {
		const auto middle = (low + high) / 2;
		if (modules[middle].end <= pc) {
			low = middle + 1;
		} else {
			high = middle;
		}
	}
	return low < count && modules[low].base <= pc ? &modules[low] : nullptr;
}

const RUNTIME_FUNCTION* FindFunction(const Module& module, uint32_t rva) {
	size_t low = 0, high = module.count;
	while (low < high) {
		const auto middle = (low + high) / 2;
		if (module.functions[middle].EndAddress <= rva) {
			low = middle + 1;
		} else {
			high = middle;
		}
	}
	return low < module.count && module.functions[low].BeginAddress <= rva ? &module.functions[low]
	                                                                       : nullptr;
}

// Runs while the target is suspended: plain memory reads and RtlVirtualUnwind only. No objects
// with destructors here (SEH).
int Unwind(CONTEXT* context, const Module* modules, size_t module_count, const Target* target,
           uint64_t* frames) {
	int count = 0;
	__try {
		while (count < MaxFrames) {
			const uint64_t pc = context->Rip;
			if (pc == 0) break;
			frames[count++]     = pc;
			const auto* module  = FindModule(modules, module_count, pc);
			if (module == nullptr) break;
			// A return address may point just past the end of its caller.
			const auto  lookup   = count == 1 ? pc : pc - 1;
			const auto* function = FindFunction(*module, static_cast<uint32_t>(lookup - module->base));
			if (function == nullptr) {
				// Leaf function: the return address is on top of the stack.
				if (context->Rsp < target->stack_low || context->Rsp + 8 > target->stack_high) break;
				context->Rip = *reinterpret_cast<const uint64_t*>(context->Rsp);
				context->Rsp += 8;
			} else {
				void*   handler_data = nullptr;
				DWORD64 establisher  = 0;
				RtlVirtualUnwind(UNW_FLAG_NHANDLER, module->base, pc,
				                 const_cast<PRUNTIME_FUNCTION>(function), context, &handler_data,
				                 &establisher, nullptr);
			}
			if (context->Rsp < target->stack_low || context->Rsp >= target->stack_high) break;
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
	return count;
}

void WaitTick(HANDLE timer, LONGLONG hundred_ns) {
	if (timer != nullptr) {
		LARGE_INTEGER due {};
		due.QuadPart = -hundred_ns;
		if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
			WaitForSingleObject(timer, INFINITE);
			return;
		}
	}
	Sleep(1);
}

uint64_t FileTime(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
}

struct SampledThread {
	HANDLE      handle = nullptr;
	uint64_t    hash   = 0; // last stack, so an unchanged one is written as a repeat line
	bool        listed = false;
	std::string name;
};

std::string ThreadName(HANDLE handle) {
	using Describe = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static const auto describe = reinterpret_cast<Describe>(reinterpret_cast<void*>(
	    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
	std::string name = "?";
	PWSTR       wide = nullptr;
	if (describe != nullptr && SUCCEEDED(describe(handle, &wide)) && wide != nullptr) {
		char buffer[256] {};
		if (wide[0] != 0 &&
		    WideCharToMultiByte(CP_UTF8, 0, wide, -1, buffer, sizeof(buffer), nullptr, nullptr) > 0) {
			name = buffer;
			std::replace(name.begin(), name.end(), '\n', ' ');
		}
		LocalFree(wide);
	}
	return name;
}

// KYTY_DIAG_SAMPLER=3: every 10 ms, each thread of the process in turn is suspended, its stack
// unwound (bounded by the memory region its stack pointer is in: guest threads run on guest
// stacks) and resumed. Guest code has no unwind tables, so a guest frame ends its stack.
// Lines: "t <tid> <name>" (again when the name changes), "u <unix ms> <tid> <frames> <addresses,
// leaf first>", "r <unix ms> <tid>" (same stack as that thread's previous sample), and every
// second "c <unix ms> <tid> <kernel 100ns> <user 100ns>" (cumulative CPU time).
[[noreturn]] void SampleAllThreads(FILE* out, HANDLE timer) {
	const DWORD self    = GetCurrentThreadId();
	const DWORD process = GetCurrentProcessId();
	auto        modules = ListModules(out);
	auto        refreshed = GetTickCount64();
	ULONGLONG   listed    = 0;
	std::unordered_map<DWORD, SampledThread> threads;
	for (;;) {
		const auto now = GetTickCount64();
		if (now - refreshed > 5000) {
			modules   = ListModules(out);
			refreshed = now;
		}
		if (now - listed >= 1000) {
			listed = now;
			for (auto& [id, thread]: threads) thread.listed = false;
			HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
			if (snapshot != INVALID_HANDLE_VALUE) {
				THREADENTRY32 entry {};
				entry.dwSize = sizeof(entry);
				for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
					if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
					auto& thread  = threads[entry.th32ThreadID];
					thread.listed = true;
					if (thread.handle == nullptr) {
						thread.handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
						                               THREAD_QUERY_INFORMATION,
						                           FALSE, entry.th32ThreadID);
					}
					if (thread.handle == nullptr) continue;
					// Guest threads name themselves after they start: look again until named.
					if (thread.name.empty() || thread.name == "?") {
						auto name = ThreadName(thread.handle);
						if (name != thread.name) {
							thread.name = std::move(name);
							std::fprintf(out, "t %lu %s\n", entry.th32ThreadID, thread.name.c_str());
						}
					}
				}
				CloseHandle(snapshot);
			}
			const auto stamp = UnixMilliseconds();
			for (auto it = threads.begin(); it != threads.end();) {
				auto& thread = it->second;
				FILETIME created {}, exited {}, kernel {}, user {};
				if (thread.listed && thread.handle != nullptr &&
				    GetThreadTimes(thread.handle, &created, &exited, &kernel, &user)) {
					std::fprintf(out, "c %" PRIu64 " %lu %" PRIu64 " %" PRIu64 "\n", stamp, it->first,
					             FileTime(kernel), FileTime(user));
				}
				if (!thread.listed) {
					if (thread.handle != nullptr) CloseHandle(thread.handle);
					it = threads.erase(it);
				} else {
					++it;
				}
			}
			std::fflush(out);
		}
		WaitTick(timer, 100000);
		const auto stamp = UnixMilliseconds();
		for (auto& [id, thread]: threads) {
			if (thread.handle == nullptr || SuspendThread(thread.handle) == static_cast<DWORD>(-1)) continue;
			uint64_t frames[MaxFrames];
			int      count = 0;
			CONTEXT  context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (GetThreadContext(thread.handle, &context)) {
				MEMORY_BASIC_INFORMATION region {};
				Target                   bounds;
				if (VirtualQuery(reinterpret_cast<const void*>(context.Rsp), &region, sizeof(region)) != 0) {
					bounds.stack_low  = reinterpret_cast<uint64_t>(region.AllocationBase);
					bounds.stack_high = reinterpret_cast<uint64_t>(region.BaseAddress) + region.RegionSize;
				}
				count = Unwind(&context, modules.data(), modules.size(), &bounds, frames);
			}
			ResumeThread(thread.handle);
			if (count == 0) continue;
			uint64_t hash = 1469598103934665603ull ^ static_cast<uint64_t>(count);
			for (int index = 0; index < count; index++) hash = (hash ^ frames[index]) * 1099511628211ull;
			if (hash == thread.hash) {
				std::fprintf(out, "r %" PRIu64 " %lu\n", stamp, id);
				continue;
			}
			thread.hash = hash;
			std::fprintf(out, "u %" PRIu64 " %lu %d", stamp, id, count);
			for (int index = 0; index < count; index++) std::fprintf(out, " %" PRIx64, frames[index]);
			std::fputc('\n', out);
		}
	}
}

void SamplerMain() {
	const char* path = std::getenv("KYTY_DIAG_SAMPLER_OUT");
	FILE*       out  = std::fopen(path != nullptr ? path : "_diag_sampler.txt", "w");
	if (out == nullptr) return;
	if (Mode() == 3) {
		std::fprintf(out, "# kyty diag sampler v4: mode 3 (all threads), period 10.0 ms\n");
		SampleAllThreads(out, CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
		                                             TIMER_ALL_ACCESS));
	}
	const LONGLONG period = Mode() == 2 ? 20000 : 5000;
	std::fprintf(out, "# kyty diag sampler v3: mode %d, period %.1f ms; s <unix ms> <frames> <addresses, leaf first>\n",
	             Mode(), static_cast<double>(period) / 10000.0);
	HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
	                                      TIMER_ALL_ACCESS);
	auto       modules   = ListModules(out);
	auto       refreshed = GetTickCount64();
	auto       flushed   = GetTickCount64();
	std::vector<uint64_t> buffer;
	buffer.reserve(FlushSamples * (MaxFrames + 2));
	size_t pending = 0;
	const auto flush = [&] {
		for (size_t index = 0; index < buffer.size();) {
			const auto stamp = buffer[index++];
			const auto count = buffer[index++];
			std::fprintf(out, "s %" PRIu64 " %" PRIu64, stamp, count);
			for (uint64_t frame = 0; frame < count; frame++) {
				std::fprintf(out, " %" PRIx64, buffer[index++]);
			}
			std::fputc('\n', out);
		}
		std::fflush(out);
		buffer.clear();
		pending = 0;
		flushed = GetTickCount64();
	};
	for (;;) {
		const auto* target = g_target.load(std::memory_order_acquire);
		if (target == nullptr) {
			if (pending != 0) flush();
			if (GetTickCount64() - refreshed > 2000) {
				modules   = ListModules(out);
				refreshed = GetTickCount64();
			}
			WaitTick(timer, 10000);
			continue;
		}
		// Whole-thread mode never leaves the target: flush and refresh between samples.
		if (pending != 0 && GetTickCount64() - flushed > 1000) flush();
		if (GetTickCount64() - refreshed > 5000) {
			modules   = ListModules(out);
			refreshed = GetTickCount64();
		}
		WaitTick(timer, period);
		if (SuspendThread(target->thread) == static_cast<DWORD>(-1)) continue;
		uint64_t frames[MaxFrames];
		int      count = 0;
		// The scope may have ended between the load and the suspend.
		if (g_target.load(std::memory_order_acquire) == target) {
			CONTEXT context {};
			context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
			if (GetThreadContext(target->thread, &context)) {
				count = Unwind(&context, modules.data(), modules.size(), target, frames);
			}
		}
		ResumeThread(target->thread);
		if (count == 0) continue;
		buffer.push_back(UnixMilliseconds());
		buffer.push_back(static_cast<uint64_t>(count));
		buffer.insert(buffer.end(), frames, frames + count);
		if (++pending == FlushSamples) flush();
	}
}

} // namespace

Scope::Scope() {
	if (Mode() == 0) return;
	static std::once_flag started;
	std::call_once(started, [] { std::thread(SamplerMain).detach(); });
	const Target* expected = nullptr;
	const bool    claimed =
	    g_target.compare_exchange_strong(expected, CurrentTarget(), std::memory_order_acq_rel);
	// Whole-thread mode: the first thread keeps the sampler; its scopes never release it.
	m_active = claimed && Mode() == 1;
}

Scope::~Scope() {
	if (m_active) g_target.store(nullptr, std::memory_order_release);
}

#else

Scope::Scope()  = default;
Scope::~Scope() = default;

#endif

} // namespace Libs::Graphics::DiagSampler
