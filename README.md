# JD Enchantment Helper

Windows desktop application for scanning Jade Dynasty inventory and running a guarded enchantment sequence.

## Current workflow

1. Select the Jade Dynasty top-level game window from the dropdown. The app refreshes its preview while it is open.
2. On the preview, enable **Область сетки инвентаря** and drag around the complete inventory grid, from the upper-left edge of its first cell to the lower-right edge of its last cell. The default logical grid is 6 columns × 8 rows. The selection estimates cell size and spacing; inspect the overlay and adjust the geometry/threshold before automating.
3. Check the amulet row/column and the first eligible item row/column in **Распознавание и калибровка → Инвентарь**.
4. The Start button is located automatically using `assets/templates/start_button_active.png`, a 71×28 crop taken from the supplied 329×222 enchantment-window capture. The matcher uses the button outline and the Cyrillic label to distinguish **Начать** from the neighboring **Купить** and **Закрыть** buttons. It then checks the saturated blue pixels inside the matched button. A manual Start ROI is available only as an override.
5. Choose 1–10 passes using the left/right arrows and press **Начать зачарование**.

## Automation sequence

- Scan the configured inventory grid and build the logical slot table.
- Verify that the configured amulet cell is occupied.
- Right-click the amulet cell, then right-click the first available occupied item cell.
- Find the Start button by template matching. Wait until the button is blue and confirm that state in two consecutive captures before left-clicking its centre.
- Wait for either supplied OK template (success or failure). Require two consecutive matches before left-clicking OK.
- Rescan the inventory. Replace the logical table with the fresh scan and compare occupied slots. Empty slots are removed from the available queue; a still-occupied item is marked attempted and is not clicked again.
- Repeat with the next unattempted occupied item until the selected pass count is reached or no eligible items remain.

The logical inventory table in the UI shows the amulet, available items, attempted items and empty/out-of-frame cells. Debug images and score maps are written to `diagnostics\`.

## Important calibration limits

The supplied reference image shows the enchantment dialog only; it does **not** show the inventory grid. Therefore the Start button has a trained template, but inventory-panel detection is not trained from this file. The inventory grid must currently be selected manually on the live preview once per application session. The grid shape defaults to 6×8 with 90% cell-to-step ratio; adjust X/Y, cell dimensions, spacing, start cell, amulet cell and occupancy threshold to match the actual game UI.

The scanner classifies cells from image texture/edge/saturation scores, not item identity. Keep the inventory stable while scanning; do not sort or move items during an automated series. If the inventory changes layout, stop and rescan/recalibrate before continuing.

The Start matcher searches the captured game window at multiple scales. The template threshold defaults to 0.68. The blue ratio threshold defaults to 0.45; the supplied active button crop measured approximately 0.94 blue-pixel coverage with the configured HSV range. Adjust these only after checking the marked rectangle on a real game frame.

The OK detector uses the supplied success/failure templates and a default confidence threshold of 0.84. If the result dialog is visually different from the references, stop and update the templates before using automation.

## Safety controls

- **F12** is the global emergency stop, polled while automation is armed even when the game has focus.
- The normal **Стоп** button stops the current sequence. **Пауза / Продолжить** suspend and resume it.
- The target-window picker is locked while the sequence is active.
- The app moves the system pointer and sends real right/left click events to the selected foreground game window. Keep that window visible, unminimized and unobstructed.
- A window/capture/input error, missing amulet, missing next item, or Start/OK timeout stops the sequence.
- Start with one pass and watch the full first cycle. Increase the pass count only after confirming that the selected inventory cells, Start button and OK button are all marked correctly.

## Build on Windows 10

Prerequisites:

1. Rust stable 1.82 or newer with the MSVC toolchain: https://rustup.rs
2. Visual Studio Build Tools (C++ workload).
3. LLVM/Clang for OpenCV Rust bindings (e.g. `choco install llvm -y`).
4. vcpkg, with OpenCV installed for `x64-windows`.

Run:

```bat
scripts\build-test.bat
```

The script checks formatting, compiles all targets, runs Release tests and creates:
`target\release\jd-enchantment-helper.exe`.

## Run

Use `run.bat` in the repository root. It adds the vcpkg OpenCV DLL folder to PATH and starts the Release executable.
