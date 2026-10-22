# NHL 27 (PPSA34063) working branch: tools and switches

This branch carries local work on getting NHL 27 (PS5, PPSA34063, v01.003.001) running well in
KytyPS5 on Windows 11 with an NVIDIA RTX 3060 Ti, on top of upstream `main`. It is a working
branch, not a pull request: several parts are experiments or diagnostics, and some come from open
upstream pull requests (#1288, #1693, #1733, #1745, #1753, #1803). The analysis and most of the code were
written with AI assistants (Claude, GLM); the game runs and tests were done by fozter.
Paths in these scripts point at `D:\PS5_GAMES\NHL27` (the game as `NHL27_PPSA34063.zar`, the
emulator checkout as `KytyPS5`).

## Scripts

| File | Purpose |
| --- | --- |
| `run_nhl_silent_tess.cmd` | Launcher with the branch's defaults (1080p output, tessellation, dense SRT walker, fault emulation, IR validator off). Any default can be overridden by setting the variable before launching. |
| `pgo_configure.cmd` | `generate` (instrumented build), `use <profdata>`, `off`: profile-guided optimization. |
| `symbolize_samples.py` | Profiles from `KYTY_DIAG_SAMPLER` captures; `--threads` and `--timeline` for all-thread captures (mode 3). |
| `cmp_spvhash.py` | SPIR-V equality and per-shader compile-phase timings between two boots. |

## Environment switches added on this branch (selection)

| Variable | Effect |
| --- | --- |
| `KYTY_SRT_DENSE=0/1` | Dense SRT walker (default 1 in the launcher). `KYTY_SRT_DENSE_VERIFY=full` runs dense and old on every refresh and compares all outputs. |
| `KYTY_FAULT_EMULATE=0/1` | Emulate GPU-thread loads from pages that only share a 4 KiB page with GPU-written data instead of draining the GPU (default 1). |
| `KYTY_GPU_INDIRECT=1` | PR #1288's GPU path for GPU-written indirect draws, extended to rect lists (off by default). In NHL 27 it was ~10% faster only while the game ran ahead of the GPU on early end-of-pipe labels, which also hung a list-walking compute shader; with those labels deferred the gain is ~2%. `KYTY_GPU_INDIRECT_RECTLIST=0`, `KYTY_GPU_INDIRECT_FLUSH=1` isolate it. |
| `KYTY_LABEL_DEFER_ALL=1`, `KYTY_LABEL_DEFER_GPU_DRAWS=1` | Experiments: publish every end-of-pipe label, or only the first one after a GPU-path indirect draw, after the GPU completed the work before it. |
| `KYTY_DIAG_SAMPLER=1/2/3`, `KYTY_DIAG_SAMPLER_OUT=<file>` | In-process sampling profiler (3 = all threads). |
