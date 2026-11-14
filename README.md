# JD Enchantment Helper

A small Rust desktop application for preparing an interface-driven Jade Dynasty
enchantment helper on Windows 10.

## Current scope

This first increment contains the desktop UI shell:

- a selector for visible, titled top-level windows (Windows only);
- a manual window-list refresh button;
- a run-count selector from 1 to 10;
- Start, Pause, and Stop controls;
- a basic application state display.

**The buttons currently change only the UI state. They do not automate the game.**
No mouse input, inventory scanning, screen capture, enchantment, or result
recognition is implemented yet.

## Build

Install the stable Rust toolchain, then run:

```powershell
cargo check
cargo test
cargo run
```

The window enumeration implementation uses Win32 APIs and is compiled only on
Windows. On other platforms the UI remains buildable, but no windows are listed.

## Planned architecture

- `app`: egui interface and user-selected settings;
- `model`: shared window and run-state types;
- `platform`: platform-specific window discovery.

Future increments can add inventory-region configuration, screenshot analysis,
and a state-driven enchantment workflow. A run means one complete pass through
the user-selected inventory region.
