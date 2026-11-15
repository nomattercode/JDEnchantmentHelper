@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not defined VCPKG_ROOT if defined VCPKG_INSTALLATION_ROOT set "VCPKG_ROOT=%VCPKG_INSTALLATION_ROOT%"
if not defined VCPKG_ROOT if exist "C:\vcpkg\installed\x64-windows\bin\opencv_core4.dll" set "VCPKG_ROOT=C:\vcpkg"

if defined VCPKG_ROOT (
    set "PATH=%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%"
)

if not exist "target\release\jd-enchantment-helper.exe" (
    echo [ERROR] Release executable not found.
    echo Run scripts\build-test.bat first.
    pause
    exit /b 1
)

start "" "target\release\jd-enchantment-helper.exe"
