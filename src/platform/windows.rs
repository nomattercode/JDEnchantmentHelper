use windows_sys::Win32::Foundation::{BOOL, LPARAM};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
    IsWindowVisible, HWND,
};

use crate::model::GameWindow;

unsafe extern "system" fn collect_window(hwnd: HWND, parameter: LPARAM) -> BOOL {
    if IsWindowVisible(hwnd) == 0 {
        return 1;
    }

    let title_length = GetWindowTextLengthW(hwnd);
    if title_length <= 0 {
        return 1;
    }

    let mut title_buffer = vec![0_u16; title_length as usize + 1];
    let copied = GetWindowTextW(
        hwnd,
        title_buffer.as_mut_ptr(),
        title_buffer.len() as i32,
    );

    if copied <= 0 {
        return 1;
    }

    let title = String::from_utf16_lossy(&title_buffer[..copied as usize]);
    let mut process_id = 0_u32;
    GetWindowThreadProcessId(hwnd, &mut process_id);

    // EnumWindows gives this callback the pointer passed by list_visible_windows.
    let windows = unsafe { &mut *(parameter as *mut Vec<GameWindow>) };
    windows.push(GameWindow {
        handle: hwnd as isize,
        title,
        process_id,
    });

    1
}

/// Returns visible top-level windows that have a non-empty title.
pub fn list_visible_windows() -> Vec<GameWindow> {
    let mut windows = Vec::new();

    // The pointer remains valid until EnumWindows returns; the callback does
    // not store it or use it asynchronously.
    unsafe {
        EnumWindows(
            Some(collect_window),
            &mut windows as *mut Vec<GameWindow> as LPARAM,
        );
    }

    windows
}
