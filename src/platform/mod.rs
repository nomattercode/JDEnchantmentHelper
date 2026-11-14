mod windows;

#[cfg(windows)]
pub use windows::list_visible_windows;

#[cfg(not(windows))]
pub fn list_visible_windows() -> Vec<crate::model::GameWindow> {
    Vec::new()
}
