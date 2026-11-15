mod windows;

#[cfg(windows)]
pub use windows::{capture_window, list_visible_windows};

#[cfg(not(windows))]
pub fn list_visible_windows() -> Vec<crate::model::GameWindow> {
    Vec::new()
}

#[cfg(not(windows))]
pub fn capture_window(_handle: isize) -> Result<opencv::core::Mat, String> {
    Err("Захват окон поддерживается только в Windows".into())
}
