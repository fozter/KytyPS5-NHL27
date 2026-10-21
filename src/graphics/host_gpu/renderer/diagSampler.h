#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGSAMPLER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGSAMPLER_H_

namespace Libs::Graphics::DiagSampler {

// Diagnostic (uncommitted): in-process sampling profiler that needs no admin rights.
// KYTY_DIAG_SAMPLER=1: while a Scope is alive, a background thread suspends the scope's thread
// about every 0.5 ms and records its call stack, unwound with the modules' own unwind tables (no
// locks are taken while the thread is suspended). KYTY_DIAG_SAMPLER=2: the first thread to enter
// a Scope (the GPU thread, at its first shader compile) is sampled every 2 ms for the rest of the
// run. KYTY_DIAG_SAMPLER=3: every thread of the process every 10 ms from the first Scope on, plus
// per-thread CPU times (guest threads are named after their guest names). Timestamped raw
// addresses go to _diag_sampler.txt in the working directory (KYTY_DIAG_SAMPLER_OUT overrides);
// tools\symbolize_samples.py builds the profile. Modes 1/2 sample one thread at a time; nested and
// concurrent scopes are ignored.
class Scope {
public:
	Scope();
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool m_active = false;
};

} // namespace Libs::Graphics::DiagSampler

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGSAMPLER_H_ */
