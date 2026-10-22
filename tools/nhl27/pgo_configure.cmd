@echo off
rem Reconfigures _Build\windows for PGO (instructions 8.26). Usage:
rem   pgo_configure.cmd generate          instrumented build (KYTY_PGO_GENERATE=ON, no ThinLTO)
rem   pgo_configure.cmd use <profdata>    optimized with the merged profile + ThinLTO
rem   pgo_configure.cmd lto               ThinLTO without a profile
rem   pgo_configure.cmd off               back to the plain build
rem Then build (cmake --build _Build/windows). Every switch rebuilds everything.
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin;D:\PS5_GAMES\NHL27\tools\glslang\bin;%PATH%"
cd /d D:\PS5_GAMES\NHL27\KytyPS5
rem A CMake re-run re-applies the zarchive patch, which fails on an already patched tree.
git -C _Build\windows\_deps\zarchive_source-src checkout -- .
if "%1"=="generate" cmake -S . -B _Build/windows -DKYTY_PGO_GENERATE=ON -DKYTY_PGO_USE= -DKYTY_THIN_LTO=OFF
if "%1"=="use" cmake -S . -B _Build/windows -DKYTY_PGO_GENERATE=OFF "-DKYTY_PGO_USE=%~2" -DKYTY_THIN_LTO=ON
if "%1"=="lto" cmake -S . -B _Build/windows -DKYTY_PGO_GENERATE=OFF -DKYTY_PGO_USE= -DKYTY_THIN_LTO=ON
if "%1"=="off" cmake -S . -B _Build/windows -DKYTY_PGO_GENERATE=OFF -DKYTY_PGO_USE= -DKYTY_THIN_LTO=OFF
if errorlevel 1 exit /b 1
echo === Configured: %* ===
endlocal
