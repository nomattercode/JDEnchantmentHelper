# JD Enchantment Helper

Windows desktop application for scanning Jade Dynasty inventory and running a guarded enchantment sequence.

## Current workflow

1. Select the Jade Dynasty top-level game window from the dropdown. The app refreshes its preview while it is open.
2. On the preview, enable **Область сетки инвентаря** and drag around the complete inventory grid, from the upper-left edge of its first cell to the lower-right edge of its last cell. The default logical grid is 6 columns × 8 rows. The selection estimates cell size and spacing; the scanner verifies that every cell stays inside this rectangle before any click is allowed.
3. Check the amulet row/column and the first eligible item row/column in **Распознавание и калибровка → Инвентарь**. Press **Сканировать инвентарь** to build and inspect the logical table without sending game input. Correct the grid geometry or occupancy threshold until the table matches the visible bag.
4. The enchantment dialog is first located from its title bar using `assets/templates/enchant_header.png`, a crop of the supplied reference capture. The last dialog rectangle is saved to `diagnostics\\enchant_window_last_seen.txt`. The Start button is then located inside that dialog using `assets/templates/start_button_active.png`, a 71×28 crop. The matcher uses the button outline and Cyrillic label to distinguish **Начать** from the neighboring **Купить** and **Закрыть** buttons, then checks saturated blue pixels inside the button. Its last rectangle is saved to `diagnostics\\start_button_last_seen.txt`. A manual Start ROI remains available as an override.
5. Choose 1–10 passes using the left/right arrows and press **Начать зачарование**. The automation performs a fresh inventory scan before starting; the preview scan is a separate no-click check.

## Automation sequence

- Scan the configured inventory grid and build the logical slot table.
- Verify that the configured amulet cell is occupied.
- Find the enchantment dialog title bar before sending any inventory clicks; if it is not found within 30 seconds, stop.
- Right-click the amulet cell, then right-click the first available occupied item cell.
- Find the Start button inside the detected enchantment dialog by template matching. Wait until the button is blue and confirm it in two consecutive captures at the same rectangle (allowing at most 3 px of capture jitter) before left-clicking its centre.
- Wait for either supplied OK template (success or failure). In the result phase, search the full captured game surface rather than only the enchantment-dialog rectangle; this supports a separate result popup or a result dialog that replaces the enchant window. Require two consecutive matches at the same rectangle (allowing at most 3 px of capture jitter) before left-clicking OK.
- Rescan the inventory. Replace the logical table with the fresh scan and compare occupied slots. Empty slots are removed from the available queue; a still-occupied item is marked attempted and is not clicked again.
- Repeat with the next unattempted occupied item until the selected pass count is reached or no eligible items remain.

The logical inventory table in the UI shows the amulet, available items, attempted items and empty/out-of-frame cells. The most recently matched Start-button rectangle is written to `diagnostics\\start_button_last_seen.txt`. Debug images and score maps are written to `diagnostics\`.

## Important calibration limits

The supplied reference image trains two templates: the enchantment title bar and the active Start button. It does **not** show the inventory grid. Therefore inventory-panel detection is not trained from this file; the complete grid must currently be selected manually on the live preview once per application session. The grid shape defaults to 6×8 with 90% cell-to-step ratio; adjust X/Y, cell dimensions, spacing, start cell, amulet cell and occupancy threshold to match the actual game UI.

The scanner classifies cells from image texture/edge/saturation scores, not item identity. Use **Сканировать инвентарь** to inspect the grid before the first run. All cells must fit inside the calibrated grid region and captured frame; an out-of-region cell blocks automation before a click. The live grid is rechecked again immediately before every right-click. Keep the inventory stable while scanning; do not sort or move items during an automated series. If the inventory changes layout, stop and rescan/recalibrate before continuing.

The enchant-header matcher searches the captured game window at multiple scales and estimates the 302×200 dialog bounds from the 299×23 title-bar crop. The Start matcher searches only inside those dialog bounds. The Start template threshold defaults to 0.68. The blue ratio threshold defaults to 0.45; the supplied active button crop measured approximately 0.94 blue-pixel coverage with the configured HSV range. Adjust these only after checking the marked rectangles on a real game frame.

The OK detector uses the supplied success/failure templates (82×42 px) and a default confidence threshold of 0.84. After Start has been clicked, the result phase searches the full captured game frame because the OK popup may be outside the enchantment dialog or may replace it. The application still requires two consecutive matches at a stable position (3 px jitter tolerance) before clicking. If the result button is visually different from the references, stop and update the templates before using automation.

## Safety controls

- **F12** is the global emergency stop, polled while automation is armed even when the game has focus.
- The normal **Стоп** button stops the current sequence. **Пауза / Продолжить** suspend and resume it.
- The target-window picker is locked while the sequence is active.
- The app moves the system pointer and sends real right/left click events to the selected foreground game window. Keep that window visible, unminimized and unobstructed.
- A window/capture/input error, missing amulet, missing next item, or Start/OK timeout stops the sequence.
- Start with one pass and watch the full first cycle. Increase the pass count only after confirming that the selected inventory cells, Start button and OK button are all marked correctly.

## Self-contained Windows build

The project uses static OpenCV libraries and a static MSVC runtime for distribution. The target PC does not need OpenCV DLLs, vcpkg, Rust, LLVM, Visual Studio Build Tools, or a separate VC++ Redistributable installation.

Build prerequisites on the **developer/build PC only**:

1. Rust stable 1.82 or newer with the `x86_64-pc-windows-msvc` toolchain: https://rustup.rs
2. Visual Studio Build Tools with the C++ workload.
3. LLVM/Clang for OpenCV Rust bindings (e.g. `choco install llvm -y`).
4. vcpkg. The script installs `opencv4:x64-windows-static` automatically if it is not already installed.

Build and test:

```bat
scripts\build-test.bat
```

The script checks formatting, checks all targets, runs Release tests, builds the statically linked executable, and creates:

- `dist\JD-Enchantment-Helper-Portable\jd-enchantment-helper.exe`
- `dist\JD-Enchantment-Helper-Portable\run.bat`
- `dist\JD-Enchantment-Helper-Portable.zip`

Copy the ZIP or the complete portable folder to another 64-bit Windows 10/11 PC and run `run.bat`. No OpenCV DLL copying or PATH configuration is required.

The static configuration is in `.cargo\config.toml` and uses `-C target-feature=+crt-static` with the vcpkg `x64-windows-static` triplet.

## Run from the source repository

Run `run.bat` in the repository root. It launches the portable executable if present beside it, otherwise it launches `target\release\jd-enchantment-helper.exe`.

GitHub Actions does not run automatically on pushes or pull requests; workflow runs are manual only.
