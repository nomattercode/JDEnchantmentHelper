@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Prefer the self-contained executable beside this launcher (portable package).
set "APP=%~dp0jd-enchantment-helper.exe"

rem When run from the source repository, use the Release build.
if not exist "%APP%" if exist "%~dp0target\release\jd-enchantment-helper.exe" set "APP=%~dp0target\release\jd-enchantment-helper.exe"

if not exist "%APP%" (
    echo [ERROR] Application executable was not found.
    echo For a source checkout, run scripts\build-test.bat first.
    echo For another PC, copy the complete JD-Enchantment-Helper-Portable folder.
    pause
    exit /b 1
)

start "" "%APP%"
