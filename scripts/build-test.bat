@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo ============================================================
echo JD Enchantment Helper - test build
echo ============================================================

where cargo >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Rust/Cargo was not found. Install Rust stable from https://rustup.rs
    exit /b 1
)

if not defined VCPKG_ROOT if exist "C:\vcpkg\vcpkg.exe" set "VCPKG_ROOT=C:\vcpkg"
if not defined VCPKG_ROOT if defined VCPKG_INSTALLATION_ROOT set "VCPKG_ROOT=%VCPKG_INSTALLATION_ROOT%"
if not defined VCPKG_ROOT (
    echo [ERROR] VCPKG_ROOT is not set and C:\vcpkg was not found.
    echo Install vcpkg and set VCPKG_ROOT to its directory.
    exit /b 1
)
if not exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo [ERROR] vcpkg.exe was not found in "%VCPKG_ROOT%".
    exit /b 1
)
where clang.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Clang was not found. Install LLVM, for example: choco install llvm -y
    echo Reopen this terminal and run this script again.
    exit /b 1
)
echo [INFO] Clang executable(s):
where clang.exe
echo [INFO] Clang version:
clang --version

set "VCPKGRS_TRIPLET=x64-windows"
set "VCPKGRS_DYNAMIC=1"
set "PATH=%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%"

rem OpenCV 4.14 from vcpkg installs headers below include\opencv4\opencv2.
rem Older ports may use include\opencv2.
set "OPENCV_INCLUDE_DIR="
if exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_ROOT%\installed\x64-windows\include"
if not defined OPENCV_INCLUDE_DIR if exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv4\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_ROOT%\installed\x64-windows\include\opencv4"

if not defined OPENCV_INCLUDE_DIR (
    echo [INFO] OpenCV headers not found. Installing OpenCV through vcpkg...
    "%VCPKG_ROOT%\vcpkg.exe" install opencv4:x64-windows
    if errorlevel 1 (
        echo [ERROR] vcpkg failed to install OpenCV.
        exit /b 1
    )
    if exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_ROOT%\installed\x64-windows\include"
    if not defined OPENCV_INCLUDE_DIR if exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv4\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_ROOT%\installed\x64-windows\include\opencv4"
)
if not defined OPENCV_INCLUDE_DIR (
    echo [ERROR] OpenCV installation completed, but core.hpp was not found.
    echo Checked:
    echo   %VCPKG_ROOT%\installed\x64-windows\include\opencv2\core.hpp
    echo   %VCPKG_ROOT%\installed\x64-windows\include\opencv4\opencv2\core.hpp
    echo Inspect the installed include tree:
    dir /s /b "%VCPKG_ROOT%\installed\x64-windows\include\core.hpp" 2^>nul
    exit /b 1
)
echo [INFO] OpenCV headers found: "%OPENCV_INCLUDE_DIR%\opencv2\core.hpp"

echo.
echo [1/4] Checking Rust formatting...
cargo fmt --all
if errorlevel 1 exit /b 1

echo.
echo [2/4] Checking all targets...
cargo check --all-targets
if errorlevel 1 exit /b 1

echo.
echo [3/4] Running tests in Release configuration (matches the app build)...
cargo test --release --all-targets
if errorlevel 1 exit /b 1

echo.
echo [4/4] Building release executable...
cargo build --release
if errorlevel 1 exit /b 1

echo.
echo [SUCCESS] Build and tests completed.
echo Executable: target\release\jd-enchantment-helper.exe
echo Keep this OpenCV folder on PATH when launching the app:
echo %VCPKG_ROOT%\installed\x64-windows\bin
exit /b 0
