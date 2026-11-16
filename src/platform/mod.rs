mod windows;

#[cfg(windows)]
pub use windows::{
    capture_window, click_window_point, f12_pressed, list_visible_windows,
    right_click_window_point,
};

#[cfg(not(windows))]
pub fn list_visible_windows() -> Vec<crate::model::GameWindow> {
    Vec::new()
}

#[cfg(not(windows))]
pub fn capture_window(_handle: isize) -> Result<opencv::core::Mat, String> {
    Err("Захват окон поддерживается только в Windows".into())
}

#[cfg(not(windows))]
pub fn click_window_point(_handle: isize, _x: i32, _y: i32) -> Result<(), String> {
    Err("Игровой ввод поддерживается только в Windows".into())
}

#[cfg(not(windows))]
pub fn f12_pressed() -> bool {
    false
}

#[cfg(not(windows))]
pub fn right_click_window_point(_handle: isize, _x: i32, _y: i32) -> Result<(), String> {
    Err("Игровой ввод поддерживается только в Windows".into())
}
