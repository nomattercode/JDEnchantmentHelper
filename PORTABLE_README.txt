JD ENCHANTMENT HELPER - PORTABLE VERSION
========================================

Run:
  Double-click run.bat
  or double-click jd-enchantment-helper.exe

This package is built for 64-bit Windows 10/11.

OpenCV and its native dependencies are statically linked into the executable.
The MSVC C/C++ runtime is also statically linked.

On the destination PC you do NOT need:
- OpenCV DLL files
- vcpkg
- Rust or Cargo
- LLVM / Clang
- Visual Studio Build Tools
- a separate Visual C++ Redistributable installation

Keep the executable and run.bat in the same folder.

The portable package is created by scripts\build-test.bat on the build PC.
