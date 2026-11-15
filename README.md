# JD Enchantment Helper

Rust desktop diagnostic application for analysing Jade Dynasty inventory and enchantment UI screenshots. Windows 10 is the primary target.

## Current functionality

- Select a visible top-level game window (selection only; live capture is not wired yet).
- Select 1–10 logical inventory passes.
- Start / Pause / Stop UI state controls. **These buttons do not send input to the game.**
- Inventory grid scanner using OpenCV:
  - configurable origin, cell size, spacing, rows and columns;
  - configurable start cell and amulet cell (amulet is always excluded from candidates);
  - grayscale variance, Canny edge density, HSV saturation, or combined occupancy score;
  - annotated overlay and per-cell score map.
- Enchantment confirmation detector using multi-scale OpenCV template matching for the blue OK button from both supplied outcomes.
- Optional estimate of the Insert button's active state using a configurable HSV ROI. Disabled by default because its coordinates must be calibrated against a full game-window screenshot.
- Debug log with each cell's coordinates, score, threshold, classification and each template-match confidence.
- Diagnostic PNGs saved to `diagnostics/` (input frame, annotated overlay, score map).
- Unicode screenshot paths are supported: image bytes are read by Rust and decoded by OpenCV, rather than relying on Windows ANSI path APIs.

No game mouse/keyboard automation, live screen capture, inventory scrolling, or automatic enchantment cycle is implemented in this increment.

## Build on Windows 10

Prerequisites:

1. Rust stable with the MSVC toolchain: https://rustup.rs
2. Visual Studio Build Tools (C++ workload).
3. LLVM/Clang (OpenCV's Rust binding generator needs Clang). For example: `choco install llvm -y`.
4. vcpkg installed, with `VCPKG_ROOT` pointing to its directory. The build script installs `opencv4:x64-windows` automatically if it is not present.

Run:

```bat
scripts\build-test.bat
```

The script performs `cargo fmt --check`, `cargo check --all-targets`, `cargo test --all-targets`, and `cargo build --release`. It places the vcpkg OpenCV DLL directory on PATH for the current script session. The executable is created at `target\release\jd-enchantment-helper.exe`.

## Run screenshot diagnostics

1. Start `jd-enchantment-helper.exe`.
2. Paste a screenshot path into **Файл изображения**. Paths with Cyrillic characters are supported.
3. Select **Инвентарь** or **Зачарование / OK**.
4. Adjust the parameters and press **Анализировать**.
5. Inspect the overlay shown in the app, the OpenCV log on the right, and PNG files in `diagnostics/`.

### Inventory tuning

Initial geometry is based on the supplied 317×478 screenshot and is only an estimate:

- origin: `(34, 84)`;
- grid: `6 columns × 8 rows`;
- cell: `36×36`;
- spacing: `40×40`;
- inner inset: `6` pixels;
- amulet cell: row 1, column 1;
- initial start: row 1, column 2.

The indices shown in the UI are 1-based. First align the rectangles with the cell frames, then adjust the detection method and threshold. Green means a candidate item, orange means the amulet, gray means before the start cell, red means an empty cell, and gray-white means the cell is outside the image. If a candidate is missed or an empty cell is marked as occupied, inspect its numeric score in the overlay/log and move the threshold or change the method.

### Enchantment detector tuning

The OK templates are crops from the supplied success and failure dialogs. The detector searches the full supplied image at multiple scales (default 0.80–1.25) and reports the best normalized correlation score. Lower the confidence threshold only after inspecting false positives. A green rectangle means the score passed the threshold; amber means a candidate match was found but did not pass.

The Insert-button HSV check is optional and off by default. The configured ROI is relative to the test image. The default values were estimated from the 324×220 cropped enchantment-dialog screenshot and are not guaranteed to align with a full client screenshot.

## Project layout

- `src/app.rs` — egui UI, diagnostic controls and preview.
- `src/vision/inventory_scanner.rs` — inventory-grid segmentation and annotated output.
- `src/vision/enchant_detector.rs` — multi-scale OK-button matching and optional Insert-button ROI check.
- `src/vision/image_io.rs` — Unicode-safe image read/write and preview conversion.
- `assets/templates/` — cropped OK-button templates from the supplied screenshots.
- `scripts/build-test.bat` — Windows test/build script.

## Safety boundary

The current application is diagnostic only. No scanner result is sent to the game, and no synthetic input is generated. Validate geometry and recognition against screenshots before any future automation layer is considered.
