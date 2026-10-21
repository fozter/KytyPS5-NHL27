@echo off
rem Builds the given CMake targets (default: kyty_emulator) with VS 2022 + clang-cl.
rem Usage: build_target.cmd [target ...]
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin;D:\PS5_GAMES\NHL27\tools\glslang\bin;%PATH%"
cd /d D:\PS5_GAMES\NHL27\KytyPS5
set "TARGETS=%*"
if "%TARGETS%"=="" set "TARGETS=kyty_emulator"
cmake --build _Build/windows --target %TARGETS% --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1
echo === Done ===
endlocal
