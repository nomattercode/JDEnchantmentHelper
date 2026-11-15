use opencv::{
    core::{self, Mat, Scalar, CV_8UC4},
    imgproc,
    prelude::*,
};
use std::{thread, time::Duration};
use windows_sys::Win32::{
    Foundation::{HWND, RECT},
    Graphics::Gdi::{
        BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDIBits,
        GetWindowDC, ReleaseDC, SelectObject, BITMAPINFO, BI_RGB, DIB_RGB_COLORS, HBITMAP, HDC,
        HGDIOBJ, SRCCOPY,
    },
    System::Threading::GetCurrentProcessId,
    UI::{
        Input::KeyboardAndMouse::{
            GetAsyncKeyState, mouse_event, MOUSEEVENTF_LEFTDOWN,
            MOUSEEVENTF_LEFTUP, VK_F12,
        },
        WindowsAndMessaging::{
            EnumWindows, GetForegroundWindow, GetWindowRect, GetWindowTextLengthW,
            GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindow, IsWindowVisible,
            SetForegroundWindow, SetCursorPos, ShowWindow, SW_RESTORE,
        },
    },
};

use crate::model::GameWindow;

unsafe extern "system" fn collect_window(hwnd: HWND, parameter: isize) -> i32 {
    if IsWindowVisible(hwnd) == 0 {
        return 1;
    }

    let title_length = GetWindowTextLengthW(hwnd);
    if title_length <= 0 {
        return 1;
    }

    let mut title_buffer = vec![0_u16; title_length as usize + 1];
    let copied = GetWindowTextW(hwnd, title_buffer.as_mut_ptr(), title_buffer.len() as i32);
    if copied <= 0 {
        return 1;
    }

    let title = String::from_utf16_lossy(&title_buffer[..copied as usize]);
    let mut process_id = 0_u32;
    GetWindowThreadProcessId(hwnd, &mut process_id);
    // Never offer this helper's own UI window as a possible game target.
    if process_id == GetCurrentProcessId() {
        return 1;
    }

    let mut bounds = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    if GetWindowRect(hwnd, &mut bounds) == 0 {
        return 1;
    }
    let width = bounds.right - bounds.left;
    let height = bounds.bottom - bounds.top;
    if width < 120 || height < 80 {
        return 1;
    }

    // EnumWindows gives this callback the pointer passed by list_visible_windows.
    let windows = unsafe { &mut *(parameter as *mut Vec<GameWindow>) };
    windows.push(GameWindow {
        handle: hwnd as isize,
        title,
        process_id,
        width: width as u32,
        height: height as u32,
    });

    1
}

/// Returns visible top-level windows with a title and useful preview dimensions.
pub fn list_visible_windows() -> Vec<GameWindow> {
    let mut windows: Vec<GameWindow> = Vec::new();

    // The pointer remains valid until EnumWindows returns; the callback does
    // not store it or use it asynchronously.
    unsafe {
        EnumWindows(
            Some(collect_window),
            &mut windows as *mut Vec<GameWindow> as isize,
        );
    }

    windows.sort_by(|a, b| {
        a.process_id
            .cmp(&b.process_id)
            .then_with(|| a.title.to_lowercase().cmp(&b.title.to_lowercase()))
            .then_with(|| a.handle.cmp(&b.handle))
    });
    windows
}

/// Captures a top-level window into a BGR OpenCV Mat using Windows GDI.
pub fn capture_window(handle: isize) -> Result<Mat, String> {
    let hwnd = handle as HWND;
    if unsafe { IsWindow(hwnd) } == 0 {
        return Err("Выбранное окно больше не существует".into());
    }
    if unsafe { IsIconic(hwnd) } != 0 {
        return Err("Игровое окно свёрнуто. Разверните его перед запуском автоматизации.".into());
    }

    let mut bounds = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    if unsafe { GetWindowRect(hwnd, &mut bounds) } == 0 {
        return Err("Не удалось получить размеры окна; возможно, оно уже закрыто".into());
    }

    let width = bounds.right - bounds.left;
    let height = bounds.bottom - bounds.top;
    if width <= 0 || height <= 0 {
        return Err("Окно свёрнуто или имеет нулевой размер".into());
    }
    let pixel_count = i64::from(width) * i64::from(height);
    if pixel_count > 40_000_000 {
        return Err(format!("Слишком большое окно для захвата: {width}x{height}"));
    }

    let window_dc: HDC = unsafe { GetWindowDC(hwnd) };
    if window_dc.is_null() {
        return Err("GetWindowDC не смог получить контекст окна".into());
    }

    let memory_dc: HDC = unsafe { CreateCompatibleDC(window_dc) };
    if memory_dc.is_null() {
        unsafe { ReleaseDC(hwnd, window_dc); }
        return Err("CreateCompatibleDC завершился ошибкой".into());
    }

    let bitmap: HBITMAP = unsafe { CreateCompatibleBitmap(window_dc, width, height) };
    if bitmap.is_null() {
        unsafe {
            DeleteDC(memory_dc);
            ReleaseDC(hwnd, window_dc);
        }
        return Err("CreateCompatibleBitmap завершился ошибкой".into());
    }

    let mut selected_object: HGDIOBJ = std::ptr::null_mut();
    let capture_result = (|| -> Result<Mat, String> {
        selected_object = unsafe { SelectObject(memory_dc, bitmap as HGDIOBJ) };
        if selected_object.is_null() {
            return Err("SelectObject не смог выбрать bitmap".into());
        }

        let copied = unsafe { BitBlt(memory_dc, 0, 0, width, height, window_dc, 0, 0, SRCCOPY) };
        if copied == 0 {
            return Err("BitBlt не смог получить изображение окна".into());
        }

        // GetDIBits requires the bitmap to be deselected from the memory DC.
        unsafe { SelectObject(memory_dc, selected_object); }
        selected_object = std::ptr::null_mut();

        let mut info: BITMAPINFO = unsafe { std::mem::zeroed() };
        info.bmiHeader.biSize = std::mem::size_of_val(&info.bmiHeader) as u32;
        info.bmiHeader.biWidth = width;
        info.bmiHeader.biHeight = -height;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB;

        let byte_count = usize::try_from(pixel_count)
            .ok()
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| "Размер буфера изображения переполнен".to_string())?;
        let mut pixels = vec![0_u8; byte_count];
        let lines = unsafe {
            GetDIBits(
                memory_dc,
                bitmap,
                0,
                height as u32,
                pixels.as_mut_ptr().cast(),
                &mut info,
                DIB_RGB_COLORS,
            )
        };
        if lines == 0 {
            return Err("GetDIBits не смог прочитать пиксели".into());
        }

        let bgra = Mat::new_rows_cols_with_default(height, width, CV_8UC4, Scalar::all(0.0))
            .map_err(|error| format!("OpenCV: {error}"))?;
        let mut bgra = bgra;
        bgra.data_bytes_mut()
            .map_err(|error| format!("OpenCV: {error}"))?
            .copy_from_slice(&pixels);

        let mut bgr = Mat::default();
        imgproc::cvt_color(
            &bgra,
            &mut bgr,
            imgproc::COLOR_BGRA2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|error| format!("OpenCV: {error}"))?;
        Ok(bgr)
    })();

    unsafe {
        if !selected_object.is_null() {
            SelectObject(memory_dc, selected_object);
        }
        DeleteObject(bitmap as HGDIOBJ);
        DeleteDC(memory_dc);
        ReleaseDC(hwnd, window_dc);
    }

    capture_result
}

/// Global F12 check. The application polls this while automation is armed.
pub fn f12_pressed() -> bool {
    unsafe { GetAsyncKeyState(VK_F12 as i32) < 0 }
}

/// Brings the selected game window forward and emits one physical-style left click.
/// Coordinates are relative to the captured top-level window rectangle.
pub fn click_window_point(handle: isize, x: i32, y: i32) -> Result<(), String> {
    let hwnd = handle as HWND;
    if unsafe { IsWindow(hwnd) } == 0 {
        return Err("Выбранное игровое окно закрыто".into());
    }

    if unsafe { IsIconic(hwnd) } != 0 {
        unsafe { ShowWindow(hwnd, SW_RESTORE); }
    }
    unsafe { SetForegroundWindow(hwnd); }
    thread::sleep(Duration::from_millis(120));

    if unsafe { GetForegroundWindow() } != hwnd {
        return Err(
            "Windows не передала фокус игровому окну. Переключитесь в игру вручную и повторите запуск."
                .into(),
        );
    }

    let mut bounds = RECT { left: 0, top: 0, right: 0, bottom: 0 };
    if unsafe { GetWindowRect(hwnd, &mut bounds) } == 0 {
        return Err("Не удалось получить координаты игрового окна".into());
    }
    let width = bounds.right - bounds.left;
    let height = bounds.bottom - bounds.top;
    if x < 0 || y < 0 || x >= width || y >= height {
        return Err(format!(
            "Точка клика ({x}, {y}) выходит за границы окна {width}x{height}"
        ));
    }

    if f12_pressed() {
        return Err("Аварийная остановка F12 перед кликом".into());
    }
    if unsafe { SetCursorPos(bounds.left + x, bounds.top + y) } == 0 {
        return Err("Windows не смогла переместить курсор к кнопке".into());
    }
    if f12_pressed() {
        return Err("Аварийная остановка F12 перед кликом".into());
    }

    unsafe { mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0); }
    thread::sleep(Duration::from_millis(55));
    unsafe { mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0); }
    Ok(())
}
