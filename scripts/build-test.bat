@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

echo ============================================================
echo JD Enchantment Helper - portable static build
echo ============================================================

where cargo >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Rust/Cargo was not found. Install Rust stable from https://rustup.rs
    exit /b 1
)

for /f "tokens=2 delims=: " %%T in ('rustc -vV ^| findstr /B "host:"') do set "RUST_HOST=%%T"
if /I not "%RUST_HOST%"=="x86_64-pc-windows-msvc" (
    echo [ERROR] This project currently builds a 64-bit Windows MSVC application.
    echo Detected Rust host: %RUST_HOST%
    echo Install the MSVC toolchain with: rustup default stable-x86_64-pc-windows-msvc
    exit /b 1
)

if not defined VCPKG_ROOT if defined VCPKG_INSTALLATION_ROOT set "VCPKG_ROOT=%VCPKG_INSTALLATION_ROOT%"
if not defined VCPKG_ROOT if exist "C:\vcpkg\vcpkg.exe" set "VCPKG_ROOT=C:\vcpkg"
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

echo [INFO] Rust host: %RUST_HOST%
echo [INFO] Clang:
clang --version
echo [INFO] VCPKG_ROOT=%VCPKG_ROOT%

rem Link OpenCV and the MSVC runtime statically.
rem VCPKGRS_DYNAMIC must be absent: vcpkg-rs treats any set value as dynamic mode.
set "VCPKGRS_TRIPLET=x64-windows-static"
set "VCPKGRS_DYNAMIC="
set "VCPKG_INSTALLED=%VCPKG_ROOT%\installed\x64-windows-static"

echo [INFO] VCPKGRS_TRIPLET=%VCPKGRS_TRIPLET%
echo [INFO] VCPKGRS_DYNAMIC is unset (static libraries)
echo [INFO] Rust MSVC CRT linkage: static, configured in .cargo\config.toml

set "NEED_OPENCV=0"
if not exist "%VCPKG_INSTALLED%\include\opencv4\opencv2\core.hpp" if not exist "%VCPKG_INSTALLED%\include\opencv2\core.hpp" set "NEED_OPENCV=1"
if not exist "%VCPKG_INSTALLED%\lib\opencv_core4.lib" set "NEED_OPENCV=1"
if not exist "%VCPKG_INSTALLED%\lib\opencv_imgproc4.lib" set "NEED_OPENCV=1"
if not exist "%VCPKG_INSTALLED%\lib\opencv_imgcodecs4.lib" set "NEED_OPENCV=1"

if "%NEED_OPENCV%"=="1" (
    echo.
    echo [INFO] Installing static OpenCV and static dependencies through vcpkg.
    echo [INFO] The first static installation can take some time.
    "%VCPKG_ROOT%\vcpkg.exe" install opencv4:x64-windows-static
    if errorlevel 1 (
        echo [ERROR] vcpkg failed to install opencv4:x64-windows-static.
        exit /b 1
    )
)

set "OPENCV_INCLUDE_DIR="
if exist "%VCPKG_INSTALLED%\include\opencv4\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_INSTALLED%\include\opencv4"
if not defined OPENCV_INCLUDE_DIR if exist "%VCPKG_INSTALLED%\include\opencv2\core.hpp" set "OPENCV_INCLUDE_DIR=%VCPKG_INSTALLED%\include"

if not defined OPENCV_INCLUDE_DIR (
    echo [ERROR] OpenCV headers were not found under "%VCPKG_INSTALLED%\include".
    exit /b 1
)
if not exist "%VCPKG_INSTALLED%\lib\opencv_core4.lib" (
    echo [ERROR] Static OpenCV core library was not found.
    exit /b 1
)
if not exist "%VCPKG_INSTALLED%\lib\opencv_imgproc4.lib" (
    echo [ERROR] Static OpenCV imgproc library was not found.
    exit /b 1
)
if not exist "%VCPKG_INSTALLED%\lib\opencv_imgcodecs4.lib" (
    echo [ERROR] Static OpenCV imgcodecs library was not found.
    exit /b 1
)

echo [INFO] Static OpenCV headers: "%OPENCV_INCLUDE_DIR%\opencv2\core.hpp"
echo [INFO] Static OpenCV libraries: "%VCPKG_INSTALLED%\lib"

echo.
echo [1/5] Checking Rust formatting...
cargo fmt --all
if errorlevel 1 exit /b 1

echo.
echo [2/5] Checking all targets with static CRT...
cargo check --all-targets
if errorlevel 1 exit /b 1

echo.
echo [3/5] Running tests in Release configuration...
cargo test --release --all-targets
if errorlevel 1 exit /b 1

echo [INFO] Tests run without administrator manifest; final app will request UAC.

echo.
echo [4/5] Building statically linked Release executable with administrator manifest...
cargo build --release --features require-admin
if errorlevel 1 exit /b 1

if not exist "target\release\jd-enchantment-helper.exe" (
    echo [ERROR] Release executable was not created.
    exit /b 1
)

echo.
echo [INFO] Verifying that the executable does not import OpenCV or MSVC runtime DLLs...
where dumpbin.exe >nul 2>nul
if not errorlevel 1 (
    dumpbin /DEPENDENTS "target\release\jd-enchantment-helper.exe" > "%TEMP%\jd-enchantment-helper-deps.txt"
    if errorlevel 1 (
        echo [ERROR] dumpbin could not inspect the Release executable.
        exit /b 1
    )
    findstr /I "opencv_ VCRUNTIME MSVCP140" "%TEMP%\jd-enchantment-helper-deps.txt" >nul
    if not errorlevel 1 (
        echo [ERROR] The executable still imports a native dependency that should be static:
        findstr /I "opencv_ VCRUNTIME MSVCP140" "%TEMP%\jd-enchantment-helper-deps.txt"
        echo Refusing to package a non-portable executable.
        del /q "%TEMP%\jd-enchantment-helper-deps.txt" >nul 2>nul
        exit /b 1
    )
    del /q "%TEMP%\jd-enchantment-helper-deps.txt" >nul 2>nul
    echo [OK] No OpenCV DLL or MSVC runtime DLL imports found.
) else (
    echo [WARN] dumpbin is not available; static linking is configured but import verification was skipped.
)

echo.
echo [5/5] Preparing portable distribution...
if exist "dist\JD-Enchantment-Helper-Portable" rmdir /s /q "dist\JD-Enchantment-Helper-Portable"
mkdir "dist\JD-Enchantment-Helper-Portable"
if errorlevel 1 exit /b 1

copy /y "target\release\jd-enchantment-helper.exe" "dist\JD-Enchantment-Helper-Portable\jd-enchantment-helper.exe" >nul
if errorlevel 1 exit /b 1
copy /y "PORTABLE_README.txt" "dist\JD-Enchantment-Helper-Portable\README.txt" >nul
if errorlevel 1 exit /b 1

if exist "dist\JD-Enchantment-Helper-Portable.zip" del /q "dist\JD-Enchantment-Helper-Portable.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\JD-Enchantment-Helper-Portable\*' -DestinationPath 'dist\JD-Enchantment-Helper-Portable.zip' -Force"
if errorlevel 1 (
    echo [WARN] ZIP creation failed; the portable folder is still available.
) else (
    echo [INFO] Portable archive: dist\JD-Enchantment-Helper-Portable.zip
)

echo.
echo [SUCCESS] Static build and tests completed.
echo.
echo The portable app is here:
echo   dist\JD-Enchantment-Helper-Portable\jd-enchantment-helper.exe
echo.
echo Copy the complete folder or ZIP to another 64-bit Windows 10/11 PC.
echo No OpenCV DLLs, vcpkg folder, Rust, LLVM or VC++ Redistributable are needed there.
exit /b 0
