@echo off
rem Builds KytyPS5 (emulator only, no Qt launcher) with VS 2022 + clang-cl.
rem Usage: build_kyty.cmd [configure|build|rebuild]

setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1

set "PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin;D:\PS5_GAMES\NHL27\tools\glslang\bin;%PATH%"

cd /d D:\PS5_GAMES\NHL27\KytyPS5
if errorlevel 1 exit /b 1

if "%1"=="rebuild" rmdir /s /q _Build\windows

if not exist "_Build\windows\build.ninja" (
  echo === Configuring ===
  cmake -S . -B _Build/windows -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DKYTY_BUILD_LAUNCHER=OFF
  if errorlevel 1 exit /b 1
)

echo === Building kyty_emulator ===
cmake --build _Build/windows --target kyty_emulator --parallel %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b 1

echo === Done ===
endlocal
