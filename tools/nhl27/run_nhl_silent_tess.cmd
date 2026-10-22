@echo off
rem Same as run_nhl_silent.cmd plus --tessellation (kPatch draws are skipped without it).
rem Config default = printf-direction Silent, so the kernel/eventQueue + AJM
rem LOGF firehose (~1 MB/s to _kyty.txt) is skipped entirely. Use this to
rem measure how much of the low gameplay fps was the logging tax, and to
rem verify that the warm _PipelineCache makes menu/load transitions fast.
rem The evidence-capturing launcher is run_nhl.cmd (unchanged).
setlocal
set "ROOT=D:\PS5_GAMES\NHL27"
set "EXEDIR=%ROOT%\KytyPS5\_Build\windows"
rem The game: the ZArchive by default (2026-10-21: 205 GB packed to 57 GB; run 122 = same load
rem times and fps as the folder, which was then removed). KYTY_GAME=<dir|zar> runs another one.
if not defined KYTY_GAME set "KYTY_GAME=%ROOT%\NHL27_PPSA34063.zar"
rem Speed defaults (2026-10-19, instructions 8.11 / 8.21): 1080p output and branch conditions
rem that take both paths. Override before launching, e.g. KYTY_VIDEO_OUT_RES=2 (4K) or
rem KYTY_SRT_BRANCH_BOTH=0. The GPU crash diag writes a report on device loss.
if not defined KYTY_VIDEO_OUT_RES set "KYTY_VIDEO_OUT_RES=1"
if not defined KYTY_SRT_BRANCH_BOTH set "KYTY_SRT_BRANCH_BOTH=1"
if not defined KYTY_GPU_CRASH_DIAG set "KYTY_GPU_CRASH_DIAG=1"
rem 2026-10-20 (instructions 8.24): skip the shader IR validator, ~15% of every first-use shader
rem compile; it never fired on NHL 27. Set KYTY_IR_VALIDATE=1 before launching when chasing a
rem shader bug (a malformed shader then stops with a message instead of rendering garbage).
if not defined KYTY_IR_VALIDATE set "KYTY_IR_VALIDATE=0"
rem 2026-10-20 (instructions 8.25): GPU-thread reads of bytes the GPU did not write are emulated
rem instead of draining the GPU (false sharing in 4 KiB pages). Set KYTY_FAULT_EMULATE=0 for the
rem old path (A/B tests, or if something looks wrong).
if not defined KYTY_FAULT_EMULATE set "KYTY_FAULT_EMULATE=1"
rem 2026-10-21 (instructions 0 item 6, our 0005): the game's Toolkit state restoration writes the
rem legacy LS resource registers 0x14a/0x14b; upstream EXITs on the write (game dies when the
rem players reach the ice). The emulator now ignores them by default (every launcher is safe);
rem only the compute tests set KYTY_SH_LEGACY_LS=0. Kept here as documentation.
if not defined KYTY_SH_LEGACY_LS set "KYTY_SH_LEGACY_LS=1"
rem 2026-10-21 (instructions 8.26, user's choice): the dense SRT walker, verified exact over ~5 M
rem refreshes (runs 97/100/102, KYTY_SRT_DENSE_VERIFY=full), ~1.5x faster per walk in play. Set
rem KYTY_SRT_DENSE=0 for the old walker.
if not defined KYTY_SRT_DENSE set "KYTY_SRT_DENSE=1"

if not exist "%EXEDIR%\kyty_emulator.exe" (
  echo kyty_emulator.exe not found under _Build\windows
  exit /b 1
)

cd /d "%EXEDIR%"
echo Running (silent, tessellation): kyty_emulator.exe --game "%KYTY_GAME%" --screen-width 1280 --screen-height 720 --tessellation %*
"%EXEDIR%\kyty_emulator.exe" --game "%KYTY_GAME%" --screen-width 1280 --screen-height 720 --tessellation %*
