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

set "VCPKGRS_TRIPLET=x64-windows"
set "VCPKGRS_DYNAMIC=1"
set "PATH=%VCPKG_ROOT%\installed\x64-windows\bin;%PATH%"

if not exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv2\core.hpp" (
    echo [INFO] Installing OpenCV through vcpkg...
    "%VCPKG_ROOT%\vcpkg.exe" install opencv4:x64-windows
    if errorlevel 1 (
        echo [ERROR] vcpkg failed to install OpenCV.
        exit /b 1
    )
)
if not exist "%VCPKG_ROOT%\installed\x64-windows\include\opencv2\core.hpp" (
    echo [ERROR] OpenCV headers are missing after installation.
    exit /b 1
)

echo.
echo [1/4] Checking Rust formatting...
cargo fmt --all -- --check
if errorlevel 1 exit /b 1

echo.
echo [2/4] Checking all targets...
cargo check --all-targets
if errorlevel 1 exit /b 1

echo.
echo [3/4] Running tests...
cargo test --all-targets
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
