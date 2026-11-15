# JD Enchantment Helper

Windows desktop utility for Jade Dynasty inventory analysis and assisted enchantment runs.

## Features

- Select a game window from a standard dropdown showing title, PID and client dimensions.
- Refresh the available windows and inspect a live preview.
- Drag a rectangle over the active blue **Start** button. The button's state is evaluated only inside this calibrated region; gray means not ready and blue means ready.
- Select 1–10 enchantment passes with left/right arrows.
- Start a run sequence: the application waits for the blue Start button, verifies it in two consecutive captures, clicks its centre, waits for the matching OK button template, verifies it twice, then clicks OK. It repeats for the requested number of runs.
- Global **F12 emergency stop** is polled while automation is armed, including while the game window has focus. The normal Stop button is also available.
- Pause and Resume controls.
- Inventory grid scanning and file-based diagnostics remain available.
- Debug PNGs are saved into `diagnostics\`.

## Important operating notes

1. The active Start-button region must be calibrated once for the selected game window. Select the window, enable **Выделить область**, then drag around the full Start button in the preview. The rectangle is saved in source-window pixel coordinates.
2. Before the first automated run, use file diagnostics or inspect the live frame to confirm the region tightly surrounds the Start button. Do not include adjacent blue controls.
3. The Start detector checks for saturated blue pixels inside that rectangle. It does not use an image template because the button changes from gray to blue when both required items are inserted.
4. The OK detector uses the supplied success/failure templates. Keep the OK button's size and visual style close to the reference screenshots; adjust the confidence threshold only after inspecting false matches.
5. Automation is sequential and state-driven; it does not click by fixed delays alone. It requires two consecutive detections before each click.
6. F12 stops the cycle immediately at the next UI poll. Use it if the game layout changes, the wrong control is highlighted, or the application behaves unexpectedly.
7. The application moves the system pointer and sends real left-click events to the selected foreground window. Do not move the game window or change its resolution while automation is running.
8. Keep the game window visible and unobstructed. Minimized, closed or inaccessible windows cause automation to stop.
9. Start with one pass and observe the first cycle before increasing the run count. The tool does not guarantee that a game server will accept an enchantment attempt.

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

## Calibration

- The Start-button region is selected interactively by dragging over the preview. Its coordinates are relative to the captured top-level game window.
- The default blue detector threshold is 22%. If the gray button is detected as active, tighten the ROI or increase the threshold. If the blue button is missed, ensure the full interior of the button is inside the ROI and adjust the threshold.
- The OK match threshold defaults to 0.84. Use a higher value if non-OK regions are incorrectly recognized.

## Safety controls

- F12: global emergency stop.
- Stop button: stop the active sequence.
- Pause: temporarily suspend polling and clicks; Resume continues in the current state.
- If the target window disappears or capture/click fails, the application stops the sequence.
