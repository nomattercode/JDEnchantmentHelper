use eframe::egui;
use opencv::{core::Size, imgproc, prelude::*};
use crate::capture_worker::{AnalysisKind, CaptureWorker};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    time::{Duration, Instant},
};

use crate::model::{GameWindow, RunState, MAX_PASSES};
use crate::platform::{
    capture_window, click_window_point, f12_pressed, list_visible_windows,
    right_click_window_point, scroll_window,
};
use crate::vision::{
    detect_inventory_grid, mat_to_rgb, scan_inventory,
    inventory_scanner::{CellDebug, CellKind, InventoryScan},
    EnchantDetectorConfig, InventoryConfig, Region,
};

const AUTOMATION_POLL_INTERVAL: Duration = Duration::from_millis(80);
const ACTION_SETTLE_DELAY: Duration = Duration::from_millis(90);
const PAGE_SETTLE_DELAY: Duration = Duration::from_millis(140);
const PAGE_CONFIRMATION_DELAY: Duration = Duration::from_millis(90);
const PAGE_RETRY_DELAY: Duration = Duration::from_millis(120);
const PAGE_CHANGE_CONFIRMATIONS: u8 = 2;
const RESULT_POLL_INTERVAL: Duration = Duration::from_millis(180);
const WINDOW_PREVIEW_MAX_WIDTH: i32 = 144;
const WINDOW_PREVIEW_MAX_HEIGHT: i32 = 84;
const LIVE_PREVIEW_MAX_WIDTH: i32 = 520;
const LIVE_PREVIEW_MAX_HEIGHT: i32 = 260;
const LIVE_PREVIEW_INTERVAL: Duration = Duration::from_millis(350);

fn window_preview_size(width: i32, height: i32) -> Size {
    let width = width.max(1);
    let height = height.max(1);
    let scale = (f64::from(WINDOW_PREVIEW_MAX_WIDTH) / f64::from(width))
        .min(f64::from(WINDOW_PREVIEW_MAX_HEIGHT) / f64::from(height))
        .min(1.0);
    Size::new(
        (f64::from(width) * scale).round().max(1.0) as i32,
        (f64::from(height) * scale).round().max(1.0) as i32,
    )
}

struct WindowPreview {
    texture: egui::TextureHandle,
    size: egui::Vec2,
}

fn same_detection_region(left: Region, right: Region) -> bool {
    left.x.abs_diff(right.x) <= 3
        && left.y.abs_diff(right.y) <= 3
        && left.width.abs_diff(right.width) <= 3
        && left.height.abs_diff(right.height) <= 3
}

fn validate_inventory_scan(
    scan: &InventoryScan,
    region: Region,
    frame: &opencv::core::Mat,
) -> Result<(), String> {
    if !region.is_configured() || region.x < 0 || region.y < 0 {
        return Err("Сетка инвентаря не настроена.".into());
    }

    let right = region
        .x
        .checked_add(region.width)
        .ok_or_else(|| "Правая граница сетки выходит за диапазон координат.".to_string())?;
    let bottom = region
        .y
        .checked_add(region.height)
        .ok_or_else(|| "Нижняя граница сетки выходит за диапазон координат.".to_string())?;

    if right > frame.cols() || bottom > frame.rows() {
        return Err("Настроенная сетка выходит за границы игрового окна.".into());
    }

    for cell in &scan.cells {
        let cell_right = cell.rect.x.saturating_add(cell.rect.width);
        let cell_bottom = cell.rect.y.saturating_add(cell.rect.height);
        if cell.kind == CellKind::OutOfFrame
            || cell.rect.x < region.x
            || cell.rect.y < region.y
            || cell_right > right
            || cell_bottom > bottom
        {
            return Err(format!(
                "Ячейка {}, {} выходит за границы сетки. Повторите настройку инвентаря.",
                cell.row + 1,
                cell.col + 1
            ));
        }
    }

    Ok(())
}

/// A fingerprint of the calibrated grid is used to detect page changes after
/// wheel input. Two identical captures at the bottom mean there is no next page.
fn inventory_page_fingerprint(
    frame: &opencv::core::Mat,
    region: Region,
) -> Result<u64, String> {
    let right = region
        .x
        .checked_add(region.width)
        .ok_or_else(|| "Правая граница сетки выходит за диапазон координат.".to_string())?;
    let bottom = region
        .y
        .checked_add(region.height)
        .ok_or_else(|| "Нижняя граница сетки выходит за диапазон координат.".to_string())?;
    if !region.is_configured()
        || region.x < 0
        || region.y < 0
        || right > frame.cols()
        || bottom > frame.rows()
    {
        return Err("Сетка инвентаря выходит за границы игрового кадра.".into());
    }

    let roi = opencv::core::Mat::roi(
        frame,
        opencv::core::Rect::new(region.x, region.y, region.width, region.height),
    )
    .map_err(|error| format!("Не удалось прочитать страницу инвентаря: {error}"))?
    .try_clone()
    .map_err(|error| format!("Не удалось подготовить снимок страницы: {error}"))?;

    // A compact average-hash is less sensitive to small animated highlights,
    // cursor-adjacent changes and capture noise than hashing every raw pixel.
    let mut gray = opencv::core::Mat::default();
    opencv::imgproc::cvt_color(
        &roi,
        &mut gray,
        opencv::imgproc::COLOR_BGR2GRAY,
        0,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .map_err(|error| format!("Не удалось преобразовать страницу в оттенки серого: {error}"))?;

    let mut thumbnail = opencv::core::Mat::default();
    opencv::imgproc::resize(
        &gray,
        &mut thumbnail,
        opencv::core::Size::new(8, 8),
        0.0,
        0.0,
        opencv::imgproc::INTER_AREA,
    )
    .map_err(|error| format!("Не удалось уменьшить снимок страницы: {error}"))?;
    let pixels = thumbnail
        .data_bytes()
        .map_err(|error| format!("Не удалось получить пиксели снимка страницы: {error}"))?;
    if pixels.len() != 64 {
        return Err("Не удалось получить корректный отпечаток страницы инвентаря.".into());
    }

    let average = pixels.iter().map(|value| u64::from(*value)).sum::<u64>() / 64;
    let mut hash = 0_u64;
    for (index, value) in pixels.iter().enumerate() {
        if u64::from(*value) >= average {
            hash |= 1_u64 << index;
        }
    }
    Ok(hash)
}

fn same_inventory_page(left: u64, right: u64) -> bool {
    (left ^ right).count_ones() <= 6
}

fn should_start_another_pass(
    current_pass: u8,
    passes_total: u8,
    surviving_items: u32,
) -> bool {
    current_pass < passes_total && surviving_items > 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutomationPhase {
    DetectEnchantDialog,
    ReturnToAmuletPage,
    CheckAmuletPage,
    InsertAmulet,
    ReturnToItemPage,
    CheckItemPage,
    InsertItem,
    WaitingForStart,
    WaitingForOk,
    ReconcileInventory,
    CheckNextPage,
    ReturnToFirstPage,
    CheckPreviousPage,
}

pub struct EnchantmentHelperApp {
    capture_worker: Option<CaptureWorker>,
    pending_capture_id: Option<u64>,
    live_frame_preview: Option<egui::TextureHandle>,
    live_frame_preview_size: egui::Vec2,
    last_live_preview_update: Option<Instant>,
    last_saved_settings: Option<String>,
    settings_dirty_since: Option<Instant>,
    last_settings_save_attempt: Option<Instant>,
    windows: Vec<GameWindow>,
    selected_window: Option<isize>,
    window_previews: HashMap<isize, WindowPreview>,
    window_preview_errors: HashMap<isize, String>,
    window_preview_queue: VecDeque<isize>,
    window_preview_batch_requested: bool,

    // A single calibration screenshot is shown only while the user draws the
    // inventory grid. The texture is dropped immediately after calibration.
    calibration_preview: Option<egui::TextureHandle>,
    calibration_preview_size: egui::Vec2,
    select_inventory_region: bool,
    region_drag_start: Option<egui::Pos2>,

    passes_total: u8,
    current_pass: u8,
    items_processed_in_pass: u32,
    surviving_items_in_pass: u32,
    page_index: usize,
    current_page_fingerprint: Option<u64>,
    unchanged_page_checks: u8,
    pending_page_fingerprint: Option<u64>,
    stable_page_change_checks: u8,
    return_scroll_remaining: usize,
    target_item_page: usize,

    run_state: RunState,
    automation_phase: Option<AutomationPhase>,
    consecutive_matches: u8,
    last_detection: Option<Region>,
    next_automation_poll: Instant,
    last_wait_log: Instant,
    phase_started: Instant,
    paused_at: Option<Instant>,

    inventory_grid_region: Region,
    logical_inventory: Vec<CellDebug>,
    processed_slots_current_page: HashSet<(u8, u8)>,
    amulet_slot: Option<CellDebug>,
    amulet_inserted: bool,
    current_item: Option<CellDebug>,

    status: String,
    summary: String,
    inventory: InventoryConfig,
    enchant: EnchantDetectorConfig,
    logs: Vec<String>,
}

impl Default for EnchantmentHelperApp {
    fn default() -> Self {
        let (saved_settings, settings_warning) = match crate::settings::load() {
            Ok(settings) => (settings, None),
            Err(error) => (crate::settings::UserSettings::default(), Some(error)),
        };
        let last_saved_settings = serde_json::to_string(&saved_settings).ok();
        let mut logs = vec!["Программа готова. F12 — аварийная остановка.".into()];
        if let Some(warning) = settings_warning { logs.push(format!("Настройки не загружены: {warning}")); }
        let capture_worker = match CaptureWorker::new() {
            Ok(worker) => Some(worker),
            Err(error) => { logs.push(error); None }
        };
        Self {
            capture_worker,
            pending_capture_id: None,
            live_frame_preview: None,
            live_frame_preview_size: egui::Vec2::ZERO,
            last_live_preview_update: None,
            last_saved_settings,
            settings_dirty_since: None,
            last_settings_save_attempt: None,
            windows: list_visible_windows(),
            selected_window: None,
            window_previews: HashMap::new(),
            window_preview_errors: HashMap::new(),
            window_preview_queue: VecDeque::new(),
            window_preview_batch_requested: false,
            calibration_preview: None,
            calibration_preview_size: egui::Vec2::ZERO,
            select_inventory_region: false,
            region_drag_start: None,

            passes_total: saved_settings.passes_total,
            current_pass: 1,
            items_processed_in_pass: 0,
            surviving_items_in_pass: 0,
            page_index: 0,
            current_page_fingerprint: None,
            unchanged_page_checks: 0,
            pending_page_fingerprint: None,
            stable_page_change_checks: 0,
            return_scroll_remaining: 0,
            target_item_page: 0,

            run_state: RunState::Idle,
            automation_phase: None,
            consecutive_matches: 0,
            last_detection: None,
            next_automation_poll: Instant::now(),
            last_wait_log: Instant::now(),
            phase_started: Instant::now(),
            paused_at: None,

            inventory_grid_region: saved_settings.inventory_grid_region,
            logical_inventory: Vec::new(),
            processed_slots_current_page: HashSet::new(),
            amulet_slot: None,
            amulet_inserted: false,
            current_item: None,

            status: "Выберите окно игры и настройте сетку инвентаря. Сначала предполагается открытой первая страница.".into(),
            summary: "Ожидание выбора игрового окна.".into(),
            inventory: saved_settings.inventory,
            enchant: saved_settings.enchant,
            logs,
        }
    }
}

impl EnchantmentHelperApp {
    fn log(&mut self, message: impl Into<String>) {
        self.logs.push(message.into());
        if self.logs.len() > 400 {
            let excess = self.logs.len() - 400;
            self.logs.drain(0..excess);
        }
    }

    fn current_user_settings(&self) -> crate::settings::UserSettings {
        crate::settings::UserSettings {
            schema_version: 1,
            passes_total: self.passes_total,
            inventory: self.inventory.clone(),
            enchant: self.enchant.clone(),
            inventory_grid_region: self.inventory_grid_region,
        }
    }

    fn persist_settings_if_needed(&mut self) {
        const SAVE_DEBOUNCE: Duration = Duration::from_millis(600);
        const SAVE_RETRY_DELAY: Duration = Duration::from_secs(5);

        let settings = self.current_user_settings();
        let Ok(snapshot) = serde_json::to_string(&settings) else {
            return;
        };
        if self.last_saved_settings.as_deref() != Some(snapshot.as_str())
            && self.settings_dirty_since.is_none()
        {
            self.settings_dirty_since = Some(Instant::now());
        }

        let debounce_elapsed = self
            .settings_dirty_since
            .is_some_and(|since| since.elapsed() >= SAVE_DEBOUNCE);
        let retry_elapsed = self
            .last_settings_save_attempt
            .map_or(true, |last| last.elapsed() >= SAVE_RETRY_DELAY);
        if !debounce_elapsed || !retry_elapsed {
            return;
        }

        self.last_settings_save_attempt = Some(Instant::now());
        match crate::settings::save(&settings) {
            Ok(()) => {
                self.last_saved_settings = Some(snapshot);
                self.settings_dirty_since = None;
                self.last_settings_save_attempt = None;
                self.log("Настройки сохранены в %APPDATA%\\JDEnchantmentHelper\\settings.json.");
            }
            Err(error) => {
                // Keep the dirty marker so the save is retried after a delay.
                self.log(format!("Не удалось сохранить настройки; повтор через 5 секунд: {error}"));
            }
        }
    }

    fn update_live_frame_preview(&mut self, frame: &opencv::core::Mat, ctx: &egui::Context) {
        if self.last_live_preview_update.is_some_and(|last| last.elapsed() < LIVE_PREVIEW_INTERVAL) { return; }
        self.last_live_preview_update = Some(Instant::now());
        let width = frame.cols().max(1);
        let height = frame.rows().max(1);
        let scale = (f64::from(LIVE_PREVIEW_MAX_WIDTH) / f64::from(width))
            .min(f64::from(LIVE_PREVIEW_MAX_HEIGHT) / f64::from(height)).min(1.0);
        let size = Size::new(
            (f64::from(width) * scale).round().max(1.0) as i32,
            (f64::from(height) * scale).round().max(1.0) as i32,
        );
        let mut thumbnail = opencv::core::Mat::default();
        if let Err(error) = imgproc::resize(frame, &mut thumbnail, size, 0.0, 0.0, imgproc::INTER_AREA) {
            self.log(format!("Не удалось подготовить превью кадра: {error}"));
            return;
        }
        let Ok((width, height, pixels)) = mat_to_rgb(&thumbnail) else { return; };
        let image = egui::ColorImage::from_rgb([width, height], &pixels);
        if let Some(texture) = &mut self.live_frame_preview {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.live_frame_preview = Some(ctx.load_texture("live-game-frame", image, egui::TextureOptions::LINEAR));
        }
        self.live_frame_preview_size = egui::vec2(width as f32, height as f32);
    }

    fn update_analysis_frame_preview(
        &mut self,
        frame: &opencv::core::Mat,
        analysis: Option<&Result<crate::vision::enchant_detector::EnchantAnalysis, String>>,
        ctx: &egui::Context,
    ) {
        let mut preview = match frame.try_clone() {
            Ok(image) => image,
            Err(_) => {
                self.update_live_frame_preview(frame, ctx);
                return;
            }
        };
        if let Some(Ok(analysis)) = analysis {
            if let Some(dialog) = analysis.dialog_window {
                let _ = imgproc::rectangle(
                    &mut preview,
                    dialog,
                    opencv::core::Scalar::new(40.0, 220.0, 80.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                );
            }
            if let Some(rect) = analysis.ok_button.rect {
                let color = if analysis.ok_button.found {
                    opencv::core::Scalar::new(0.0, 220.0, 255.0, 0.0)
                } else {
                    opencv::core::Scalar::new(0.0, 0.0, 255.0, 0.0)
                };
                let _ = imgproc::rectangle(
                    &mut preview,
                    rect,
                    color,
                    2,
                    imgproc::LINE_8,
                    0,
                );
            }
            if let Some(start) = &analysis.start_button {
                let color = if start.active {
                    opencv::core::Scalar::new(255.0, 160.0, 0.0, 0.0)
                } else {
                    opencv::core::Scalar::new(120.0, 160.0, 255.0, 0.0)
                };
                let _ = imgproc::rectangle(
                    &mut preview,
                    start.rect,
                    color,
                    2,
                    imgproc::LINE_8,
                    0,
                );
            }
        }
        self.update_live_frame_preview(&preview, ctx);
    }

    fn update_inventory_detection_preview(
        &mut self,
        frame: &opencv::core::Mat,
        detection: &crate::vision::InventoryDetection,
        ctx: &egui::Context,
    ) {
        let mut preview = match frame.try_clone() {
            Ok(image) => image,
            Err(_) => {
                self.update_live_frame_preview(frame, ctx);
                return;
            }
        };
        let _ = imgproc::rectangle(
            &mut preview,
            detection.header_rect,
            opencv::core::Scalar::new(0.0, 220.0, 255.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        );
        let grid = opencv::core::Rect::new(
            detection.grid_region.x,
            detection.grid_region.y,
            detection.grid_region.width,
            detection.grid_region.height,
        );
        let _ = imgproc::rectangle(
            &mut preview,
            grid,
            opencv::core::Scalar::new(60.0, 220.0, 80.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        );
        self.last_live_preview_update = None;
        self.update_live_frame_preview(&preview, ctx);
    }

    fn detect_inventory_automatically(&mut self, ctx: &egui::Context) {
        let Some(handle) = self.selected_window else {
            self.status = "Сначала выберите игровое окно.".into();
            return;
        };
        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = format!("Не удалось получить кадр для поиска инвентаря: {error}");
                self.log(self.status.clone());
                return;
            }
        };
        self.update_live_frame_preview(&frame, ctx);
        let detection = match detect_inventory_grid(&frame, &self.inventory) {
            Ok(Some(detection)) => detection,
            Ok(None) => {
                self.status = "Заголовок инвентаря не найден с достаточной уверенностью. Используйте ручную настройку.".into();
                self.log(self.status.clone());
                return;
            }
            Err(error) => {
                self.status = format!("Ошибка поиска заголовка инвентаря: {error}");
                self.log(self.status.clone());
                return;
            }
        };
        let scan = match scan_inventory(&frame, &detection.config) {
            Ok(scan) => scan,
            Err(error) => {
                self.status = format!("Заголовок найден, но сетку не удалось проверить: {error}");
                self.log(self.status.clone());
                return;
            }
        };
        if let Err(error) = validate_inventory_scan(&scan, detection.grid_region, &frame) {
            self.status = format!("Найденная сетка не прошла проверку: {error}");
            self.log(self.status.clone());
            return;
        }
        self.update_inventory_detection_preview(&frame, &detection, ctx);
        self.inventory = detection.config;
        self.inventory_grid_region = detection.grid_region;
        self.calibration_preview = None;
        self.select_inventory_region = false;
        self.region_drag_start = None;
        self.logical_inventory = scan.cells;
        self.status = format!(
            "Инвентарь найден по заголовку (совпадение {:.0}%). Сетка {}×{} настроена автоматически.",
            detection.score * 100.0, self.inventory.columns, self.inventory.rows
        );
        self.log(self.status.clone());
    }

    fn reset_detection_streak(&mut self) {
        self.consecutive_matches = 0;
        self.last_detection = None;
    }

    fn update_detection_streak(&mut self, current: Option<Region>) -> bool {
        if let Some(current) = current {
            let same_location = self
                .last_detection
                .is_some_and(|previous| same_detection_region(previous, current));
            self.consecutive_matches = if same_location {
                self.consecutive_matches.saturating_add(1)
            } else {
                1
            };
            self.last_detection = Some(current);
        } else {
            self.reset_detection_streak();
        }
        self.consecutive_matches >= 2
    }

    fn selected_window_data(&self) -> Option<&GameWindow> {
        self.selected_window
            .and_then(|handle| self.windows.iter().find(|window| window.handle == handle))
    }

    fn selected_label(&self) -> String {
        self.selected_window_data()
            .map(|window| {
                format!(
                    "{} · {}×{}",
                    window.title, window.width, window.height
                )
            })
            .unwrap_or_else(|| "Выберите окно игры…".into())
    }

    fn request_window_previews(&mut self) {
        self.window_previews.clear();
        self.window_preview_errors.clear();
        self.window_preview_queue = self
            .windows
            .iter()
            .map(|window| window.handle)
            .collect();
        self.window_preview_batch_requested = true;
    }

    fn capture_next_window_preview(&mut self, ctx: &egui::Context) {
        let Some(handle) = self.window_preview_queue.pop_front() else {
            return;
        };

        let result = capture_window(handle).and_then(|frame| {
            let size = window_preview_size(frame.cols(), frame.rows());
            let mut thumbnail = opencv::core::Mat::default();
            imgproc::resize(
                &frame,
                &mut thumbnail,
                size,
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )
            .map_err(|error| format!("Не удалось уменьшить миниатюру окна: {error}"))?;
            let (width, height, pixels) =
                mat_to_rgb(&thumbnail).map_err(|error| error.to_string())?;
            Ok((width, height, pixels))
        });

        match result {
            Ok((width, height, pixels)) => {
                let image = egui::ColorImage::from_rgb([width, height], &pixels);
                let texture = ctx.load_texture(
                    format!("game-window-preview-{handle}"),
                    image,
                    egui::TextureOptions::LINEAR,
                );
                self.window_previews.insert(
                    handle,
                    WindowPreview {
                        texture,
                        size: egui::vec2(width as f32, height as f32),
                    },
                );
                self.window_preview_errors.remove(&handle);
            }
            Err(error) => {
                self.window_preview_errors.insert(handle, error);
            }
        }

        if !self.window_preview_queue.is_empty() {
            ctx.request_repaint_after(Duration::from_millis(25));
        }
    }

    fn refresh_windows(&mut self) {
        self.windows = list_visible_windows();
        self.request_window_previews();
        let valid: HashSet<isize> =
            self.windows.iter().map(|window| window.handle).collect();

        if self.selected_window.is_some_and(|handle| !valid.contains(&handle)) {
            self.selected_window = None;
            self.calibration_preview = None;
            self.inventory_grid_region = Region::default();
            self.logical_inventory.clear();
            if self.automation_phase.is_some() {
                self.stop_automation("Игровое окно закрыто. Автоматизация остановлена.");
            }
        }

        self.log(format!("Список окон обновлён. Найдено окон: {}.", self.windows.len()));
    }

    fn capture_calibration_snapshot(&mut self, ctx: &egui::Context) {
        let Some(handle) = self.selected_window else {
            self.status = "Сначала выберите игровое окно.".into();
            return;
        };

        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.calibration_preview = None;
                self.status = format!("Не удалось получить кадр игрового окна: {error}");
                self.log(format!("Ошибка получения кадра: {error}"));
                return;
            }
        };
        self.last_live_preview_update = None;
        self.update_live_frame_preview(&frame, ctx);
        match mat_to_rgb(&frame).map_err(|error| error.to_string()) {
            Ok((width, height, pixels)) => {
                let image = egui::ColorImage::from_rgb([width, height], &pixels);
                self.calibration_preview_size = egui::vec2(width as f32, height as f32);
                self.calibration_preview = Some(ctx.load_texture(
                    "inventory-calibration",
                    image,
                    egui::TextureOptions::LINEAR,
                ));
                self.select_inventory_region = true;
                self.status = "Кадр получен. Обведите всю сетку инвентаря от первой до последней ячейки.".into();
                self.log(format!("Получен разовый кадр для настройки: {}×{}.", width, height));
            }
            Err(error) => {
                self.calibration_preview = None;
                self.status = format!("Не удалось получить кадр игрового окна: {error}");
                self.log(format!("Ошибка получения кадра: {error}"));
            }
        }
    }

    fn scan_inventory_snapshot(
        &mut self,
        frame: &opencv::core::Mat,
        initial: bool,
    ) -> Result<(), String> {
        let mut config = self.inventory.clone();
        if self.page_index > 0 {
            // Амулет расположен на первой странице. На последующих страницах
            // все ячейки являются обычными слотами предметов.
            config.amulet_row = u8::MAX;
            config.amulet_col = u8::MAX;
            config.start_row = 0;
            config.start_col = 0;
        }

        let scan = scan_inventory(frame, &config)
            .map_err(|error| format!("Не удалось просканировать страницу инвентаря: {error}"))?;
        validate_inventory_scan(&scan, self.inventory_grid_region, frame)?;

        self.logical_inventory = scan.cells.clone();
        self.current_page_fingerprint =
            Some(inventory_page_fingerprint(frame, self.inventory_grid_region)?);

        if self.page_index == 0 && !self.amulet_inserted {
            let amulet = self.logical_inventory.iter().find(|cell| {
                cell.row == self.inventory.amulet_row && cell.col == self.inventory.amulet_col
            });
            match amulet {
                Some(cell) if cell.score >= self.inventory.threshold => {
                    self.amulet_slot = Some(cell.clone());
                }
                Some(cell) => {
                    self.amulet_slot = None;
                    return Err(format!(
                        "Амулет не распознан в ячейке {}, {}. Оценка {:.2}, порог {:.2}.",
                        cell.row + 1,
                        cell.col + 1,
                        cell.score,
                        self.inventory.threshold
                    ));
                }
                None => {
                    self.amulet_slot = None;
                    return Err("Не найдена настроенная ячейка амулета.".into());
                }
            }
        }

        let item_count = self
            .logical_inventory
            .iter()
            .filter(|cell| cell.kind == CellKind::Occupied)
            .count();
        self.summary = format!(
            "Проход {}/{} · страница {} · предметов на странице: {} · обработано: {} · уцелело: {}",
            self.current_pass,
            self.passes_total,
            self.page_index + 1,
            item_count,
            self.items_processed_in_pass,
            self.surviving_items_in_pass
        );

        self.log(format!(
            "{} страница {}: найдено занятых ячеек {}.",
            if initial { "Начальное сканирование:" } else { "Обновление:" },
            self.page_index + 1,
            item_count
        ));
        Ok(())
    }

    fn refresh_current_page_fingerprint(
        &mut self,
        frame: &opencv::core::Mat,
    ) -> Result<(), String> {
        self.current_page_fingerprint =
            Some(inventory_page_fingerprint(frame, self.inventory_grid_region)?);
        Ok(())
    }

    fn next_unprocessed_item(&self) -> Option<CellDebug> {
        self.logical_inventory
            .iter()
            .find(|cell| {
                cell.kind == CellKind::Occupied
                    && !self
                        .processed_slots_current_page
                        .contains(&(cell.row, cell.col))
            })
            .cloned()
    }

    fn prepare_amulet_for_current_item(&mut self) {
        self.target_item_page = self.page_index;
        self.amulet_inserted = false;
        self.automation_phase = Some(AutomationPhase::ReturnToAmuletPage);
        self.phase_started = Instant::now();
        self.next_automation_poll = Instant::now();
        self.log(format!(
            "Перед зачарованием предмета со страницы {} повторно выбираю амулет.",
            self.target_item_page + 1
        ));
    }

    fn begin_scroll_down(&mut self, handle: isize) -> Result<(), String> {
        let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
        let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
        scroll_window(handle, x, y, -120)?;
        self.unchanged_page_checks = 0;
        self.pending_page_fingerprint = None;
        self.stable_page_change_checks = 0;
        self.automation_phase = Some(AutomationPhase::CheckNextPage);
        self.phase_started = Instant::now();
        self.next_automation_poll = Instant::now() + PAGE_SETTLE_DELAY;
        self.status = format!(
            "Проход {} из {} · прокрутка вниз после страницы {}.",
            self.current_pass,
            self.passes_total,
            self.page_index + 1
        );
        self.log(format!("Прокручиваю инвентарь вниз от страницы {}.", self.page_index + 1));
        Ok(())
    }

    fn start_next_pass(&mut self, handle: isize, frame: &opencv::core::Mat) {
        self.current_pass = self.current_pass.saturating_add(1);
        self.items_processed_in_pass = 0;
        self.surviving_items_in_pass = 0;
        self.page_index = 0;
        self.processed_slots_current_page.clear();
        self.current_item = None;
        self.amulet_inserted = false;
        self.reset_detection_streak();

        if let Err(error) = self.scan_inventory_snapshot(frame, false) {
            self.stop_automation(&format!(
                "Не удалось начать следующий проход с первой страницы: {error}"
            ));
            return;
        }

        self.log(format!(
            "Начат проход {} из {}. Инвентарь возвращён на первую страницу.",
            self.current_pass, self.passes_total
        ));
        if let Some(item) = self.next_unprocessed_item() {
            self.current_item = Some(item.clone());
            self.prepare_amulet_for_current_item();
            self.status = format!(
                "Проход {} из {}: следующий предмет — ячейка {}, {}.",
                self.current_pass,
                self.passes_total,
                item.row + 1,
                item.col + 1
            );
        } else if let Err(error) = self.begin_scroll_down(handle) {
            self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
        }
    }

    fn finish_pass_at_bottom(&mut self, handle: isize, frame: &opencv::core::Mat) {
        let pages = self.page_index + 1;
        self.log(format!(
            "Проход {} завершён: страниц {}, обработано {}, уцелело {}.",
            self.current_pass,
            pages,
            self.items_processed_in_pass,
            self.surviving_items_in_pass
        ));

        if !should_start_another_pass(
            self.current_pass,
            self.passes_total,
            self.surviving_items_in_pass,
        ) {
            let reason = if self.surviving_items_in_pass == 0 && self.current_pass < self.passes_total {
                format!(
                    "Работа завершена досрочно: после прохода {} уцелевших предметов не осталось.",
                    self.current_pass
                )
            } else {
                format!(
                    "Готово: выполнено {} полных проходов по {} страницам инвентаря.",
                    self.current_pass, pages
                )
            };
            self.stop_automation(&reason);
            return;
        }

        self.return_scroll_remaining = self.page_index;
        if self.return_scroll_remaining == 0 {
            self.start_next_pass(handle, frame);
        } else {
            self.automation_phase = Some(AutomationPhase::ReturnToFirstPage);
            self.phase_started = Instant::now();
            self.next_automation_poll = Instant::now();
            self.status = format!(
                "Проход {} завершён. Возвращаю инвентарь на первую страницу.",
                self.current_pass
            );
            self.log(format!(
                "Для возврата на первую страницу требуется прокруток вверх: {}.",
                self.return_scroll_remaining
            ));
        }
    }

    fn start_automation(&mut self, ctx: &egui::Context) {
        if matches!(self.run_state, RunState::Running | RunState::Paused) || self.pending_capture_id.is_some() {
            return;
        }
        let Some(_handle) = self.selected_window else {
            self.status = "Сначала выберите игровое окно.".into();
            return;
        };
        if !self.inventory_grid_region.is_configured() {
            self.status = "Сначала снимите кадр и настройте сетку инвентаря.".into();
            return;
        }

        // Stop collecting one-off window thumbnails before automation starts.
        self.window_preview_queue.clear();
        self.window_preview_batch_requested = false;

        if self.capture_worker.is_none() {
            self.status = "Поток захвата кадров не запущен. Перезапустите приложение.".into();
            return;
        }
        let handle = self.selected_window.expect("Игровое окно проверено перед запуском");
        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = format!("Не удалось получить начальный кадр игры: {error}");
                return;
            }
        };

        self.update_live_frame_preview(&frame, ctx);
        self.current_pass = 1;
        self.items_processed_in_pass = 0;
        self.surviving_items_in_pass = 0;
        self.page_index = 0;
        self.current_page_fingerprint = None;
        self.unchanged_page_checks = 0;
        self.pending_page_fingerprint = None;
        self.stable_page_change_checks = 0;
        self.return_scroll_remaining = 0;
        self.paused_at = None;
        self.reset_detection_streak();
        self.processed_slots_current_page.clear();
        self.current_item = None;
        self.amulet_slot = None;
        self.amulet_inserted = false;
        self.logical_inventory.clear();

        if let Err(error) = self.scan_inventory_snapshot(&frame, true) {
            self.run_state = RunState::Stopped;
            self.automation_phase = None;
            self.status = error.clone();
            self.log(format!("Не удалось начать серию: {error}"));
            return;
        }

        self.automation_phase = Some(AutomationPhase::DetectEnchantDialog);
        self.run_state = RunState::Running;
        self.phase_started = Instant::now();
        self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
        self.last_wait_log = Instant::now();
        self.status = format!(
            "Запущен проход 1 из {}. Поиск окна зачарования. Начальная страница — первая.",
            self.passes_total
        );
        self.log(format!(
            "Начата серия: {} полных проходов по всем страницам инвентаря. Начинаю с первой страницы.",
            self.passes_total
        ));
    }

    fn stop_automation(&mut self, reason: &str) {
        let was_active = self.automation_phase.is_some()
            || matches!(self.run_state, RunState::Running | RunState::Paused);
        self.automation_phase = None;
        self.paused_at = None;
        self.reset_detection_streak();
        self.run_state = RunState::Stopped;
        self.status = reason.into();
        if was_active {
            self.log(format!("ОСТАНОВКА: {reason}"));
        }
    }

    fn update_automation(&mut self, ctx: &egui::Context) {
        if self.automation_phase.is_some() && f12_pressed() {
            self.stop_automation("Аварийная остановка: нажата F12.");
        }

        // Keep at most one frame in flight. Capture and OpenCV matching run on
        // a worker thread; only completed image data is copied into the UI.
        if self.pending_capture_id.is_none() {
            if self.run_state != RunState::Running || self.automation_phase.is_none() { return; }
            let now = Instant::now();
            if now < self.next_automation_poll { return; }
            let Some(handle) = self.selected_window else {
                self.stop_automation("Игровое окно не выбрано.");
                return;
            };
            let analysis_kind = match self.automation_phase {
                Some(AutomationPhase::DetectEnchantDialog | AutomationPhase::WaitingForStart) => AnalysisKind::EnchantDialog,
                Some(AutomationPhase::WaitingForOk) => AnalysisKind::ResultPopup,
                _ => AnalysisKind::None,
            };
            let Some(worker) = self.capture_worker.as_mut() else {
                self.stop_automation("Поток захвата кадров недоступен.");
                return;
            };
            match worker.request(handle, analysis_kind, self.enchant.clone()) {
                Ok(request_id) => {
                    self.pending_capture_id = Some(request_id);
                    self.next_automation_poll = now + AUTOMATION_POLL_INTERVAL;
                }
                Err(error) => {
                    self.stop_automation(&format!("Не удалось поставить кадр в очередь: {error}"));
                    return;
                }
            }
            ctx.request_repaint_after(Duration::from_millis(20));
            return;
        }

        let response = match self.capture_worker.as_ref().map(CaptureWorker::try_receive) {
            Some(Ok(Some(response))) => response,
            Some(Ok(None)) => {
                ctx.request_repaint_after(Duration::from_millis(20));
                return;
            }
            Some(Err(error)) => {
                self.pending_capture_id = None;
                self.stop_automation(&error);
                return;
            }
            None => {
                self.pending_capture_id = None;
                self.stop_automation("Поток захвата кадров не создан.");
                return;
            }
        };
        let expected_id = self.pending_capture_id.take();
        if expected_id != Some(response.id) {
            self.log("Получен устаревший кадр; результат отброшен.");
            return;
        }
        if self.run_state != RunState::Running || self.automation_phase.is_none() { return; }
        let captured = match response.result {
            Ok(captured) => captured,
            Err(error) => {
                self.stop_automation(&format!("Не удалось получить кадр игры: {error}"));
                return;
            }
        };
        let frame = match captured.to_mat() {
            Ok(frame) => frame,
            Err(error) => {
                self.stop_automation(&error);
                return;
            }
        };
        let mut capture_analysis = captured.analysis;
        self.update_analysis_frame_preview(&frame, capture_analysis.as_ref(), ctx);
        let now = Instant::now();
        let Some(handle) = self.selected_window else {
            self.stop_automation("Игровое окно не выбрано.");
            return;
        };

        match self.automation_phase {
            Some(AutomationPhase::DetectEnchantDialog) => {
                let analysis = match capture_analysis.take() {
                    Some(Ok(analysis)) => analysis,
                    Some(Err(error)) => {
                        self.stop_automation(&format!("Ошибка поиска окна зачарования: {error}"));
                        return;
                    }
                    None => {
                        self.stop_automation("Рабочий поток не вернул результат поиска окна зачарования.");
                        return;
                    }
                };
                if analysis.dialog_window.is_some() {
                    self.log("Окно зачарования найдено. Подготавливаю первый предмет.");
                    self.reset_detection_streak();
                    self.current_item = self.next_unprocessed_item();
                    if self.current_item.is_some() {
                        self.prepare_amulet_for_current_item();
                    } else if let Err(error) = self.begin_scroll_down(handle) {
                        self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                    }
                } else if now.duration_since(self.phase_started) >= Duration::from_secs(30) {
                    self.stop_automation("Окно зачарования не найдено за 30 секунд. Клики не выполнялись.");
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("Ожидание окна зачарования.");
                }
                return;
            }
            Some(AutomationPhase::InsertAmulet) => {
                if self.page_index != 0 {
                    self.stop_automation("Для выбора амулета нужно вернуться на первую страницу инвентаря.");
                    return;
                }
                if let Err(error) = self.scan_inventory_snapshot(&frame, false) {
                    self.stop_automation(&format!("Не удалось подтвердить амулет: {error}"));
                    return;
                }
                let Some(amulet) = self.amulet_slot.clone() else {
                    self.stop_automation("Ячейка амулета не настроена.");
                    return;
                };
                let x = amulet.rect.x + amulet.rect.width / 2;
                let y = amulet.rect.y + amulet.rect.height / 2;
                match right_click_window_point(handle, x, y) {
                    Ok(()) => {
                        self.amulet_inserted = true;
                        self.log(format!(
                            "Амулет выбран из ячейки {}, {}.",
                            amulet.row + 1,
                            amulet.col + 1
                        ));
                        if self.page_index < self.target_item_page {
                            self.automation_phase = Some(AutomationPhase::ReturnToItemPage);
                            self.phase_started = Instant::now();
                            self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                            self.status = format!(
                                "Амулет выбран. Возвращаюсь на страницу {} за предметом.",
                                self.target_item_page + 1
                            );
                        } else {
                            self.automation_phase = Some(AutomationPhase::InsertItem);
                            self.phase_started = Instant::now();
                            self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                            self.status = "Амулет помещён в окно зачарования. Выбираю предмет.".into();
                        }
                        self.reset_detection_streak();
                    }
                    Err(error) => self.stop_automation(&format!("Не удалось выбрать амулет: {error}")),
                }
                return;
            }
            Some(AutomationPhase::InsertItem) => {
                let Some(item) = self.current_item.clone().or_else(|| self.next_unprocessed_item()) else {
                    if let Err(error) = self.begin_scroll_down(handle) {
                        self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                    }
                    return;
                };

                if let Err(error) = self.scan_inventory_snapshot(&frame, false) {
                    self.stop_automation(&format!("Не удалось перепроверить ячейку предмета: {error}"));
                    return;
                }
                let verified = self.logical_inventory
                    .iter()
                    .find(|cell| cell.row == item.row && cell.col == item.col)
                    .cloned();
                let Some(verified) = verified else {
                    self.stop_automation("Выбранная ячейка отсутствует на текущей странице.");
                    return;
                };

                if verified.kind != CellKind::Occupied || verified.score < self.inventory.threshold {
                    self.processed_slots_current_page.insert((item.row, item.col));
                    self.items_processed_in_pass = self.items_processed_in_pass.saturating_add(1);
                    self.current_item = self.next_unprocessed_item();
                    self.log(format!(
                        "Ячейка {}, {} уже пуста. Пропускаю её.",
                        item.row + 1,
                        item.col + 1
                    ));
                    if self.current_item.is_none() {
                        if let Err(error) = self.begin_scroll_down(handle) {
                            self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                        }
                    } else {
                        self.phase_started = Instant::now();
                        self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                    }
                    return;
                }

                self.current_item = Some(verified.clone());
                let x = verified.rect.x + verified.rect.width / 2;
                let y = verified.rect.y + verified.rect.height / 2;
                match right_click_window_point(handle, x, y) {
                    Ok(()) => {
                        self.log(format!(
                            "Выбран предмет: страница {}, ячейка {}, {}.",
                            self.page_index + 1,
                            verified.row + 1,
                            verified.col + 1
                        ));
                        self.automation_phase = Some(AutomationPhase::WaitingForStart);
                        self.reset_detection_streak();
                        self.phase_started = Instant::now();
                        self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                        self.status = format!(
                            "Проход {} из {} · страница {} · предмет {}, {}. Ожидание кнопки «Начать».",
                            self.current_pass,
                            self.passes_total,
                            self.page_index + 1,
                            verified.row + 1,
                            verified.col + 1
                        );
                    }
                    Err(error) => self.stop_automation(&format!("Не удалось выбрать предмет: {error}")),
                }
                return;
            }
            Some(AutomationPhase::ReconcileInventory) => {
                let current_item = self.current_item.clone();
                if let Err(error) = self.scan_inventory_snapshot(&frame, false) {
                    self.stop_automation(&format!("Не удалось обновить инвентарь после зачарования: {error}"));
                    return;
                }

                if let Some(previous) = current_item {
                    self.processed_slots_current_page.insert((previous.row, previous.col));
                    self.items_processed_in_pass = self.items_processed_in_pass.saturating_add(1);
                    let updated = self.logical_inventory
                        .iter()
                        .find(|cell| cell.row == previous.row && cell.col == previous.col);
                    match updated.map(|cell| cell.kind) {
                        Some(CellKind::Occupied) => {
                            self.surviving_items_in_pass = self.surviving_items_in_pass.saturating_add(1);
                            self.log(format!(
                                "Предмет в ячейке {}, {} уцелел. Он будет доступен в следующем проходе.",
                                previous.row + 1,
                                previous.col + 1
                            ));
                        }
                        Some(CellKind::Empty) => self.log(format!(
                            "Предмет в ячейке {}, {} исчез после зачарования.",
                            previous.row + 1,
                            previous.col + 1
                        )),
                        _ => self.log(format!(
                            "Состояние ячейки {}, {} после зачарования не подтверждено.",
                            previous.row + 1,
                            previous.col + 1
                        )),
                    }
                }

                self.current_item = self.next_unprocessed_item();
                if let Some(item) = self.current_item.clone() {
                    self.prepare_amulet_for_current_item();
                    self.status = format!(
                        "Проход {} из {} · страница {} · следующий предмет: {}, {}.",
                        self.current_pass,
                        self.passes_total,
                        self.page_index + 1,
                        item.row + 1,
                        item.col + 1
                    );
                } else if let Err(error) = self.begin_scroll_down(handle) {
                    self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                }
                return;
            }
            Some(AutomationPhase::CheckNextPage) => {
                let fingerprint = match inventory_page_fingerprint(&frame, self.inventory_grid_region) {
                    Ok(value) => value,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось проверить страницу инвентаря: {error}"));
                        return;
                    }
                };

                if now.duration_since(self.phase_started) > Duration::from_secs(10) {
                    self.stop_automation("Не удалось надёжно определить, переключилась ли страница инвентаря.");
                    return;
                }

                if self.current_page_fingerprint.is_some_and(|previous| same_inventory_page(previous, fingerprint)) {
                    self.pending_page_fingerprint = None;
                    self.stable_page_change_checks = 0;
                    if self.unchanged_page_checks == 0 {
                        self.unchanged_page_checks = 1;
                        self.next_automation_poll = Instant::now() + PAGE_RETRY_DELAY;
                        return;
                    }
                    self.log(format!("Следующая страница после {} не обнаружена.", self.page_index + 1));
                    self.finish_pass_at_bottom(handle, &frame);
                    return;
                }

                self.unchanged_page_checks = 0;
                let stable_candidate = self
                    .pending_page_fingerprint
                    .is_some_and(|previous| same_inventory_page(previous, fingerprint));
                if stable_candidate {
                    self.stable_page_change_checks = self.stable_page_change_checks.saturating_add(1);
                } else {
                    self.pending_page_fingerprint = Some(fingerprint);
                    self.stable_page_change_checks = 1;
                }
                if self.stable_page_change_checks < PAGE_CHANGE_CONFIRMATIONS {
                    self.next_automation_poll = Instant::now() + PAGE_CONFIRMATION_DELAY;
                    return;
                }

                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.page_index = self.page_index.saturating_add(1);
                self.processed_slots_current_page.clear();
                self.current_item = None;
                if let Err(error) = self.scan_inventory_snapshot(&frame, false) {
                    self.stop_automation(&format!(
                        "Не удалось просканировать страницу {}: {error}",
                        self.page_index + 1
                    ));
                    return;
                }

                self.log(format!("Открыта страница {}.", self.page_index + 1));
                self.current_item = self.next_unprocessed_item();
                if self.current_item.is_some() {
                    self.prepare_amulet_for_current_item();
                } else if let Err(error) = self.begin_scroll_down(handle) {
                    self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                }
                return;
            }
            Some(AutomationPhase::ReturnToAmuletPage) => {
                if self.page_index == 0 {
                    if let Err(error) = self.scan_inventory_snapshot(&frame, false) {
                        self.stop_automation(&format!("Не удалось проверить амулет на первой странице: {error}"));
                        return;
                    }
                    self.automation_phase = Some(AutomationPhase::InsertAmulet);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                    return;
                }

                if let Err(error) = self.refresh_current_page_fingerprint(&frame) {
                    self.stop_automation(&format!("Не удалось обновить отпечаток текущей страницы: {error}"));
                    return;
                }
                let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
                let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
                if let Err(error) = scroll_window(handle, x, y, 120) {
                    self.stop_automation(&format!("Не удалось перейти к амулету на первой странице: {error}"));
                    return;
                }
                self.unchanged_page_checks = 0;
                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.automation_phase = Some(AutomationPhase::CheckAmuletPage);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + PAGE_SETTLE_DELAY;
                self.status = format!(
                    "Возвращаюсь к амулету. Текущая страница: {}.",
                    self.page_index + 1
                );
                return;
            }
            Some(AutomationPhase::CheckAmuletPage) => {
                let fingerprint = match inventory_page_fingerprint(&frame, self.inventory_grid_region) {
                    Ok(value) => value,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось проверить страницу при поиске амулета: {error}"));
                        return;
                    }
                };

                if now.duration_since(self.phase_started) > Duration::from_secs(10) {
                    self.stop_automation("Не удалось надёжно определить переход к первой странице за амулетом.");
                    return;
                }

                if self.current_page_fingerprint.is_some_and(|previous| same_inventory_page(previous, fingerprint)) {
                    self.pending_page_fingerprint = None;
                    self.stable_page_change_checks = 0;
                    if self.unchanged_page_checks == 0 {
                        self.unchanged_page_checks = 1;
                        self.next_automation_poll = Instant::now() + PAGE_RETRY_DELAY;
                        return;
                    }
                    self.stop_automation("Колесо мыши не переключило страницу вверх при поиске амулета.");
                    return;
                }

                self.unchanged_page_checks = 0;
                let stable_candidate = self
                    .pending_page_fingerprint
                    .is_some_and(|previous| same_inventory_page(previous, fingerprint));
                if stable_candidate {
                    self.stable_page_change_checks = self.stable_page_change_checks.saturating_add(1);
                } else {
                    self.pending_page_fingerprint = Some(fingerprint);
                    self.stable_page_change_checks = 1;
                }
                if self.stable_page_change_checks < PAGE_CHANGE_CONFIRMATIONS {
                    self.next_automation_poll = Instant::now() + PAGE_CONFIRMATION_DELAY;
                    return;
                }

                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.current_page_fingerprint = Some(fingerprint);
                self.page_index = self.page_index.saturating_sub(1);
                self.automation_phase = Some(AutomationPhase::ReturnToAmuletPage);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                return;
            }
            Some(AutomationPhase::ReturnToItemPage) => {
                if self.page_index >= self.target_item_page {
                    self.automation_phase = Some(AutomationPhase::InsertItem);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                    return;
                }

                if let Err(error) = self.refresh_current_page_fingerprint(&frame) {
                    self.stop_automation(&format!("Не удалось обновить отпечаток текущей страницы: {error}"));
                    return;
                }
                let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
                let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
                if let Err(error) = scroll_window(handle, x, y, -120) {
                    self.stop_automation(&format!("Не удалось вернуться к странице предмета: {error}"));
                    return;
                }
                self.unchanged_page_checks = 0;
                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.automation_phase = Some(AutomationPhase::CheckItemPage);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + PAGE_SETTLE_DELAY;
                self.status = format!(
                    "Возвращаюсь к предметам. Текущая страница: {} из {}.",
                    self.page_index + 1,
                    self.target_item_page + 1
                );
                return;
            }
            Some(AutomationPhase::CheckItemPage) => {
                let fingerprint = match inventory_page_fingerprint(&frame, self.inventory_grid_region) {
                    Ok(value) => value,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось проверить страницу при возврате к предмету: {error}"));
                        return;
                    }
                };

                if now.duration_since(self.phase_started) > Duration::from_secs(10) {
                    self.stop_automation("Не удалось надёжно определить возврат на страницу предмета.");
                    return;
                }

                if self.current_page_fingerprint.is_some_and(|previous| same_inventory_page(previous, fingerprint)) {
                    self.pending_page_fingerprint = None;
                    self.stable_page_change_checks = 0;
                    if self.unchanged_page_checks == 0 {
                        self.unchanged_page_checks = 1;
                        self.next_automation_poll = Instant::now() + PAGE_RETRY_DELAY;
                        return;
                    }
                    self.stop_automation("Колесо мыши не переключило страницу вниз при возврате к предмету.");
                    return;
                }

                self.unchanged_page_checks = 0;
                let stable_candidate = self
                    .pending_page_fingerprint
                    .is_some_and(|previous| same_inventory_page(previous, fingerprint));
                if stable_candidate {
                    self.stable_page_change_checks = self.stable_page_change_checks.saturating_add(1);
                } else {
                    self.pending_page_fingerprint = Some(fingerprint);
                    self.stable_page_change_checks = 1;
                }
                if self.stable_page_change_checks < PAGE_CHANGE_CONFIRMATIONS {
                    self.next_automation_poll = Instant::now() + PAGE_CONFIRMATION_DELAY;
                    return;
                }

                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.current_page_fingerprint = Some(fingerprint);
                self.page_index = self.page_index.saturating_add(1);
                self.automation_phase = Some(AutomationPhase::ReturnToItemPage);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                return;
            }
            Some(AutomationPhase::ReturnToFirstPage) => {
                if self.return_scroll_remaining == 0 {
                    self.start_next_pass(handle, &frame);
                    return;
                }

                if let Err(error) = self.refresh_current_page_fingerprint(&frame) {
                    self.stop_automation(&format!("Не удалось обновить отпечаток текущей страницы: {error}"));
                    return;
                }
                let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
                let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
                if let Err(error) = scroll_window(handle, x, y, 120) {
                    self.stop_automation(&format!("Не удалось вернуться к предыдущей странице: {error}"));
                    return;
                }
                self.unchanged_page_checks = 0;
                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.automation_phase = Some(AutomationPhase::CheckPreviousPage);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + PAGE_SETTLE_DELAY;
                self.status = format!(
                    "Возврат на первую страницу. Осталось переходов: {}.",
                    self.return_scroll_remaining
                );
                return;
            }
            Some(AutomationPhase::CheckPreviousPage) => {
                let fingerprint = match inventory_page_fingerprint(&frame, self.inventory_grid_region) {
                    Ok(value) => value,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось проверить предыдущую страницу: {error}"));
                        return;
                    }
                };

                if now.duration_since(self.phase_started) > Duration::from_secs(10) {
                    self.stop_automation("Не удалось надёжно подтвердить возврат к первой странице.");
                    return;
                }

                let unchanged = self
                    .current_page_fingerprint
                    .is_some_and(|previous| same_inventory_page(previous, fingerprint));
                if unchanged {
                    self.pending_page_fingerprint = None;
                    self.stable_page_change_checks = 0;
                    if self.unchanged_page_checks == 0 {
                        self.unchanged_page_checks = 1;
                        self.next_automation_poll = Instant::now() + PAGE_RETRY_DELAY;
                        return;
                    }
                    self.stop_automation(
                        "Колесо мыши не переключило страницу вверх. Следующий проход не запущен.",
                    );
                    return;
                }

                self.unchanged_page_checks = 0;
                let stable_candidate = self
                    .pending_page_fingerprint
                    .is_some_and(|previous| same_inventory_page(previous, fingerprint));
                if stable_candidate {
                    self.stable_page_change_checks = self.stable_page_change_checks.saturating_add(1);
                } else {
                    self.pending_page_fingerprint = Some(fingerprint);
                    self.stable_page_change_checks = 1;
                }
                if self.stable_page_change_checks < PAGE_CHANGE_CONFIRMATIONS {
                    self.next_automation_poll = Instant::now() + PAGE_CONFIRMATION_DELAY;
                    return;
                }

                self.pending_page_fingerprint = None;
                self.stable_page_change_checks = 0;
                self.current_page_fingerprint = Some(fingerprint);
                self.page_index = self.page_index.saturating_sub(1);
                self.return_scroll_remaining = self.return_scroll_remaining.saturating_sub(1);
                self.log(format!(
                    "Подтверждён переход вверх. Текущая страница: {}.",
                    self.page_index + 1
                ));

                if self.return_scroll_remaining == 0 {
                    self.start_next_pass(handle, &frame);
                } else {
                    self.automation_phase = Some(AutomationPhase::ReturnToFirstPage);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                }
                return;
            }
            Some(AutomationPhase::WaitingForStart)
            | Some(AutomationPhase::WaitingForOk) => {}
            None => return,
        }

        let phase = self.automation_phase;
        let timeout = match phase {
            Some(AutomationPhase::DetectEnchantDialog) => Duration::from_secs(30),
            Some(AutomationPhase::WaitingForStart) => Duration::from_secs(45),
            Some(AutomationPhase::WaitingForOk) => Duration::from_secs(180),
            Some(AutomationPhase::CheckNextPage) => Duration::from_secs(8),
            Some(AutomationPhase::CheckPreviousPage) => Duration::from_secs(8),
            Some(AutomationPhase::ReturnToFirstPage) => Duration::from_secs(8),
            _ => Duration::from_secs(30),
        };
        if now.duration_since(self.phase_started) > timeout {
            let message = match phase {
                Some(AutomationPhase::WaitingForStart) => "Не дождался активации кнопки «Начать».",
                Some(AutomationPhase::WaitingForOk) => "Не дождался появления кнопки «ОК».",
                Some(AutomationPhase::CheckNextPage) => "Не удалось определить результат прокрутки инвентаря.",
                Some(AutomationPhase::CheckPreviousPage) => "Не удалось подтвердить переход на предыдущую страницу.",
                Some(AutomationPhase::ReturnToFirstPage) => "Не удалось вернуться к первой странице инвентаря.",
                _ => "Истекло время ожидания этапа автоматизации.",
            };
            self.stop_automation(message);
            return;
        }

        let analysis = match capture_analysis.take() {
            Some(Ok(analysis)) => analysis,
            Some(Err(error)) => {
                self.stop_automation(&format!("Ошибка распознавания окна зачарования: {error}"));
                return;
            }
            None => {
                self.stop_automation("Рабочий поток не вернул результат распознавания.");
                return;
            }
        };

        if let Some(button) = &analysis.start_button {
            self.summary = format!(
                "Проход {}/{} · страница {} · кнопка «Начать»: {}",
                self.current_pass,
                self.passes_total,
                self.page_index + 1,
                if button.active { "активна" } else { "неактивна" }
            );
        }

        match phase {
            Some(AutomationPhase::WaitingForStart) => {
                let current = analysis.start_button.as_ref()
                    .filter(|button| button.active)
                    .map(|button| Region {
                        x: button.rect.x,
                        y: button.rect.y,
                        width: button.rect.width,
                        height: button.rect.height,
                    });
                if self.update_detection_streak(current) {
                    if f12_pressed() {
                        self.stop_automation("Аварийная остановка: нажата F12.");
                        return;
                    }
                    let Some(button) = analysis.start_button else {
                        self.reset_detection_streak();
                        return;
                    };
                    let x = button.rect.x + button.rect.width / 2;
                    let y = button.rect.y + button.rect.height / 2;
                    match click_window_point(handle, x, y) {
                        Ok(()) => {
                            self.log("Кнопка «Начать» подтверждена и нажата.");
                            self.automation_phase = Some(AutomationPhase::WaitingForOk);
                            self.reset_detection_streak();
                            self.phase_started = Instant::now();
                            self.next_automation_poll = Instant::now() + RESULT_POLL_INTERVAL;
                            self.status = format!(
                                "Проход {} из {} · страница {} · зачарование запущено. Ожидание «ОК».",
                                self.current_pass, self.passes_total, self.page_index + 1
                            );
                        }
                        Err(error) => self.stop_automation(&format!("Не удалось нажать «Начать»: {error}")),
                    }
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("Ожидание синей кнопки «Начать».");
                }
            }
            Some(AutomationPhase::WaitingForOk) => {
                self.summary = format!(
                    "Поиск кнопки «ОК»: совпадение {:.0}% · требуемый порог {:.0}%.",
                    analysis.ok_button.score * 100.0,
                    self.enchant.confidence_threshold * 100.0
                );
                let current = if analysis.ok_button.found {
                    analysis.ok_button.rect.map(|rect| Region {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    })
                } else {
                    None
                };
                if self.update_detection_streak(current) {
                    if f12_pressed() {
                        self.stop_automation("Аварийная остановка: нажата F12.");
                        return;
                    }
                    let Some(rect) = analysis.ok_button.rect else {
                        self.reset_detection_streak();
                        return;
                    };
                    let x = rect.x + rect.width / 2;
                    let y = rect.y + rect.height / 2;
                    match click_window_point(handle, x, y) {
                        Ok(()) => {
                            self.log("Кнопка «ОК» подтверждена и нажата. Обновляю текущую страницу инвентаря.");
                            self.reset_detection_streak();
                            self.automation_phase = Some(AutomationPhase::ReconcileInventory);
                            self.phase_started = Instant::now();
                            self.next_automation_poll = Instant::now() + ACTION_SETTLE_DELAY;
                        }
                        Err(error) => self.stop_automation(&format!("Не удалось нажать «ОК»: {error}")),
                    }
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("Ожидание кнопки «ОК».");
                }
            }
            _ => {}
        }
    }

    fn inventory_settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.label("Размер сетки");
        let previous_columns = self.inventory.columns;
        let previous_rows = self.inventory.rows;
        ui.horizontal(|ui| {
            ui.label("Столбцы");
            ui.add(egui::DragValue::new(&mut self.inventory.columns).range(1..=12));
            ui.label("Строки");
            ui.add(egui::DragValue::new(&mut self.inventory.rows).range(1..=30));
        });
        if self.inventory.columns != previous_columns || self.inventory.rows != previous_rows {
            self.inventory_grid_region = Region::default();
            self.calibration_preview = None;
            self.select_inventory_region = false;
            self.logical_inventory.clear();
            self.status = "Размер сетки изменён. Повторите настройку инвентаря.".into();
        }

        ui.separator();
        ui.label("Положение амулета на первой странице (нумерация с 1)");
        ui.horizontal(|ui| {
            ui.label("Строка");
            let mut row = self.inventory.amulet_row + 1;
            ui.add(egui::DragValue::new(&mut row).range(1..=self.inventory.rows.max(1)));
            self.inventory.amulet_row = row.saturating_sub(1).min(self.inventory.rows.saturating_sub(1));
            ui.label("Столбец");
            let mut col = self.inventory.amulet_col + 1;
            ui.add(egui::DragValue::new(&mut col).range(1..=self.inventory.columns.max(1)));
            self.inventory.amulet_col = col.saturating_sub(1).min(self.inventory.columns.saturating_sub(1));
        });

        ui.label("Первая ячейка с предметами (нумерация с 1)");
        ui.horizontal(|ui| {
            ui.label("Строка");
            let mut row = self.inventory.start_row + 1;
            ui.add(egui::DragValue::new(&mut row).range(1..=self.inventory.rows.max(1)));
            self.inventory.start_row = row.saturating_sub(1).min(self.inventory.rows.saturating_sub(1));
            ui.label("Столбец");
            let mut col = self.inventory.start_col + 1;
            ui.add(egui::DragValue::new(&mut col).range(1..=self.inventory.columns.max(1)));
            self.inventory.start_col = col.saturating_sub(1).min(self.inventory.columns.saturating_sub(1));
        });

        let mut inventory_percent = (self.inventory.threshold.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut inventory_percent, 0..=100).text("Чувствительность предметов").suffix("%")).changed() {
            self.inventory.threshold = f64::from(inventory_percent) / 100.0;
        }
        let mut inventory_header_percent = (self.inventory.header_threshold.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut inventory_header_percent, 40..=98).text("Поиск заголовка инвентаря").suffix("%")).changed() {
            self.inventory.header_threshold = f64::from(inventory_header_percent) / 100.0;
        }
        ui.separator();
        let mut ok_percent = (self.enchant.confidence_threshold.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut ok_percent, 40..=98).text("Чувствительность кнопки «ОК»").suffix("%")).changed() {
            self.enchant.confidence_threshold = f64::from(ok_percent) / 100.0;
        }
        let mut enchant_header_percent = (self.enchant.header_threshold.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut enchant_header_percent, 40..=98).text("Поиск окна зачарования").suffix("%")).changed() {
            self.enchant.header_threshold = f64::from(enchant_header_percent) / 100.0;
        }
        let mut start_percent = (self.enchant.start_template_threshold.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut start_percent, 40..=98).text("Поиск кнопки «Начать»").suffix("%")).changed() {
            self.enchant.start_template_threshold = f64::from(start_percent) / 100.0;
        }
        let mut blue_percent = (self.enchant.min_blue_ratio.clamp(0.0, 1.0) * 100.0).round() as u32;
        if ui.add(egui::Slider::new(&mut blue_percent, 10..=90).text("Порог синего цвета кнопки").suffix("%")).changed() {
            self.enchant.min_blue_ratio = f64::from(blue_percent) / 100.0;
        }

        if self.inventory_grid_region.is_configured() {
            ui.label(format!(
                "Сетка настроена: {}×{} пикс., шаг {}×{} пикс.",
                self.inventory_grid_region.width,
                self.inventory_grid_region.height,
                self.inventory.step_x,
                self.inventory.step_y
            ));
        } else {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Сетка не настроена. Нажмите «Снять кадр и настроить сетку».",
            );
        }
    }

    fn game_window_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let selection_enabled = self.automation_phase.is_none();
            let previous_window = self.selected_window;
            ui.add_enabled_ui(selection_enabled, |ui| {
                let combo = egui::ComboBox::from_id_salt("game_window")
                    .width((ui.available_width() - 90.0).max(180.0))
                    .height(420.0)
                    .selected_text(self.selected_label())
                    .show_ui(ui, |ui| {
                        for window in self.windows.clone() {
                            ui.horizontal(|ui| {
                                if let Some(preview) = self.window_previews.get(&window.handle) {
                                    let scale = (112.0 / preview.size.x)
                                        .min(66.0 / preview.size.y)
                                        .min(1.0);
                                    ui.add(egui::Image::new((
                                        preview.texture.id(),
                                        preview.size * scale,
                                    )));
                                } else if self.window_preview_queue.contains(&window.handle) {
                                    ui.allocate_ui(egui::vec2(112.0, 58.0), |ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.small("Снимок…");
                                        });
                                    });
                                } else if self.window_preview_errors.contains_key(&window.handle) {
                                    ui.allocate_ui(egui::vec2(112.0, 58.0), |ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.small("Нет превью");
                                        });
                                    });
                                } else {
                                    ui.allocate_ui(egui::vec2(112.0, 58.0), |ui| {
                                        ui.centered_and_justified(|ui| {
                                            ui.small("Превью не снято");
                                        });
                                    });
                                }

                                let label = format!(
                                    "{} · {}×{}",
                                    window.title, window.width, window.height
                                );
                                ui.selectable_value(
                                    &mut self.selected_window,
                                    Some(window.handle),
                                    label,
                                );
                            });
                            ui.separator();
                        }
                    });
                if combo.inner.is_some() && !self.window_preview_batch_requested {
                    self.request_window_previews();
                }
                if ui.button("↻").on_hover_text("Обновить список окон и миниатюры").clicked() {
                    self.refresh_windows();
                }
            });

            if self.selected_window != previous_window {
                self.calibration_preview = None;
                self.select_inventory_region = false;
                if previous_window.is_some() {
                    self.inventory_grid_region = Region::default();
                }
                self.logical_inventory.clear();
                self.processed_slots_current_page.clear();
                self.amulet_slot = None;
                self.amulet_inserted = false;
                self.current_item = None;
                self.page_index = 0;
                self.current_page_fingerprint = None;
                self.status = if self.inventory_grid_region.is_configured() {
                    "Игровое окно выбрано. Сохранённая геометрия сетки загружена; проверьте её перед запуском.".into()
                } else {
                    "Игровое окно выбрано. Откройте первую страницу инвентаря и найдите сетку.".into()
                };
                if previous_window.is_some() {
                    self.log("Выбрано другое игровое окно. Геометрия старой сетки сброшена.");
                } else {
                    self.log("Игровое окно выбрано. Сохранённые параметры распознавания загружены.");
                }
            }
        });

    }

    fn inventory_calibration_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if self.selected_window_data().is_none() {
            ui.label("Сначала выберите игровое окно слева.");
            return;
        }

        ui.horizontal(|ui| {
            ui.add_enabled_ui(self.automation_phase.is_none(), |ui| {
                if ui.button("Найти инвентарь автоматически").clicked() {
                    self.detect_inventory_automatically(ctx);
                }
                if ui.button("Снять кадр и выделить сетку").clicked() {
                    self.capture_calibration_snapshot(ctx);
                }
            });
            if self.calibration_preview.is_some() {
                ui.label("Выделите мышью всю сетку — от первой до последней ячейки.");
            } else if self.inventory_grid_region.is_configured() {
                ui.label("Сетка настроена. Повторная настройка не требуется.");
            }
        });

        if let Some(texture) = self.calibration_preview.as_ref() {
            let texture_id = texture.id();
            let raw_size = texture.size_vec2();
            let scale = (ui.available_width() / raw_size.x.max(1.0))
                .min(520.0 / raw_size.y.max(1.0))
                .min(1.0);
            let image_size = raw_size * scale;
            let response = ui.add(
                egui::Image::new((texture_id, image_size))
                    .sense(if self.select_inventory_region {
                        egui::Sense::click_and_drag()
                    } else {
                        egui::Sense::hover()
                    }),
            );

            if self.select_inventory_region && response.drag_started() {
                self.region_drag_start = response.interact_pointer_pos();
            }
            if self.select_inventory_region && response.dragged() && self.region_drag_start.is_none() {
                self.region_drag_start = response.interact_pointer_pos();
            }
            if self.select_inventory_region && response.drag_stopped() {
                let start = self.region_drag_start.take();
                let end = response
                    .interact_pointer_pos()
                    .or_else(|| ctx.input(|input| input.pointer.interact_pos()));
                if let (Some(start), Some(end)) = (start, end) {
                    let clamp_point = |point: egui::Pos2| {
                        egui::pos2(
                            point.x.clamp(response.rect.left(), response.rect.right()),
                            point.y.clamp(response.rect.top(), response.rect.bottom()),
                        )
                    };
                    let start = clamp_point(start);
                    let end = clamp_point(end);
                    let to_image = |point: egui::Pos2| {
                        let x = ((point.x - response.rect.left()) / response.rect.width()) * raw_size.x;
                        let y = ((point.y - response.rect.top()) / response.rect.height()) * raw_size.y;
                        (x.round() as i32, y.round() as i32)
                    };
                    let (x1, y1) = to_image(egui::pos2(start.x.min(end.x), start.y.min(end.y)));
                    let (x2, y2) = to_image(egui::pos2(start.x.max(end.x), start.y.max(end.y)));

                    if x2 - x1 >= 12 && y2 - y1 >= 8 {
                        let width = x2 - x1;
                        let height = y2 - y1;
                        let columns = f64::from(self.inventory.columns.max(1));
                        let rows = f64::from(self.inventory.rows.max(1));
                        let step_x = (f64::from(width) / (columns - 1.0 + 0.9)).floor() as i32;
                        let step_y = (f64::from(height) / (rows - 1.0 + 0.9)).floor() as i32;

                        if step_x < 8 || step_y < 8 {
                            self.status = "Выбранная область слишком мала для такого количества ячеек.".into();
                        } else {
                            self.inventory.origin_x = x1;
                            self.inventory.origin_y = y1;
                            self.inventory.step_x = step_x;
                            self.inventory.step_y = step_y;
                            self.inventory.cell_width =
                                ((f64::from(step_x) * 0.9).floor() as i32).clamp(8, step_x);
                            self.inventory.cell_height =
                                ((f64::from(step_y) * 0.9).floor() as i32).clamp(8, step_y);
                            self.inventory_grid_region = Region {
                                x: x1,
                                y: y1,
                                width,
                                height,
                            };
                            self.select_inventory_region = false;
                            self.calibration_preview = None;
                            self.region_drag_start = None;
                            self.status = format!(
                                "Сетка настроена: {} столбцов × {} строк. Превью удалено из памяти.",
                                self.inventory.columns, self.inventory.rows
                            );
                            self.log(format!(
                                "Сетка настроена: {} столбцов × {} строк.",
                                self.inventory.columns, self.inventory.rows
                            ));
                        }
                    } else {
                        self.status = "Область слишком мала. Выделите всю сетку.".into();
                    }
                }
            }

            ui.small(format!(
                "Размер снимка: {}×{} пикселей.",
                raw_size.x as u32, raw_size.y as u32
            ));
        }
    }
}

impl eframe::App for EnchantmentHelperApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_automation(ctx);
        self.capture_next_window_preview(ctx);

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Jade Dynasty · Зачарование");
                ui.separator();
                let (state, color) = match self.run_state {
                    RunState::Running => ("В РАБОТЕ", egui::Color32::from_rgb(70, 190, 125)),
                    RunState::Paused => ("ПАУЗА", egui::Color32::from_rgb(225, 175, 65)),
                    RunState::Stopped => ("ОСТАНОВЛЕНО", egui::Color32::from_rgb(180, 115, 115)),
                    RunState::Idle if self.selected_window_data().is_none() => {
                        ("ВЫБЕРИТЕ ИГРОВОЕ ОКНО", ui.visuals().weak_text_color())
                    }
                    RunState::Idle if !self.inventory_grid_region.is_configured() => {
                        ("НАСТРОЙТЕ ИНВЕНТАРЬ", ui.visuals().weak_text_color())
                    }
                    RunState::Idle => ("ГОТОВО К ЗАПУСКУ", egui::Color32::from_rgb(70, 190, 125)),
                };
                ui.label(
                    egui::RichText::new(state)
                        .strong()
                        .color(color),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.colored_label(
                        egui::Color32::LIGHT_RED,
                        egui::RichText::new("F12 · АВАРИЙНЫЙ СТОП").strong(),
                    );
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(2, |columns| {
                    columns[0].vertical(|ui| {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("1. ИГРОВОЕ ОКНО").strong());
                            ui.add_space(4.0);
                            self.game_window_ui(ui);
                        });

                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("2. ИНВЕНТАРЬ").strong());
                            ui.add_space(4.0);
                            self.inventory_calibration_ui(ui, ctx);
                            ui.separator();

                            let grid_ready = self.inventory_grid_region.is_configured();
                            let selected = self.selected_window_data().is_some();

                            if !selected {
                                ui.label("Сначала выберите окно Jade Dynasty выше.");
                            } else if !grid_ready {
                                ui.label("Снимите кадр игры и выделите всю сетку инвентаря.");
                            } else {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("Сетка настроена")
                                            .strong()
                                            .color(egui::Color32::from_rgb(70, 190, 125)),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.small(format!(
                                                "{} × {} · {} ячеек",
                                                self.inventory.columns,
                                                self.inventory.rows,
                                                u16::from(self.inventory.columns)
                                                    * u16::from(self.inventory.rows)
                                            ));
                                        },
                                    );
                                });
                            }

                            egui::CollapsingHeader::new("Параметры сетки и распознавания")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.add_enabled_ui(self.automation_phase.is_none(), |ui| {
                                        self.inventory_settings_ui(ui);
                                    });
                                    if self.automation_phase.is_some() {
                                        ui.small("Параметры заблокированы на время серии.");
                                    }
                                });
                        });

                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("ПЕРЕД ЗАПУСКОМ").strong());
                            ui.small("Откройте первую страницу инвентаря, убедитесь, что амулет на своём месте, и не перемещайте предметы вручную во время серии.");
                        });
                    });

                    columns[1].vertical(|ui| {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("3. ЗАПУСК И УПРАВЛЕНИЕ").strong());
                            ui.add_space(4.0);

                            ui.label("Полных проходов по всем страницам");
                            ui.horizontal(|ui| {
                                let can_edit = self.automation_phase.is_none()
                                    && self.run_state != RunState::Paused
                                    && self.run_state != RunState::Running;
                                if ui.add_enabled(
                                    can_edit && self.passes_total > 1,
                                    egui::Button::new("−").min_size(egui::vec2(38.0, 34.0)),
                                ).clicked() {
                                    self.passes_total = self.passes_total.saturating_sub(1).max(1);
                                }
                                ui.label(
                                    egui::RichText::new(self.passes_total.to_string())
                                        .size(26.0)
                                        .strong()
                                        .monospace(),
                                );
                                if ui.add_enabled(
                                    can_edit && self.passes_total < MAX_PASSES,
                                    egui::Button::new("+").min_size(egui::vec2(38.0, 34.0)),
                                ).clicked() {
                                    self.passes_total = self.passes_total.saturating_add(1).min(MAX_PASSES);
                                }
                                ui.small(format!("1–{MAX_PASSES}"));
                            });
                            ui.small("Один проход — полный обход всех страниц. Следующие проходы затрагивают только уцелевшие предметы.");

                            ui.add_space(8.0);
                            let can_start = self.automation_phase.is_none()
                                && self.run_state != RunState::Running
                                && self.run_state != RunState::Paused
                                && self.selected_window.is_some()
                                && self.inventory_grid_region.is_configured()
                                && self.pending_capture_id.is_none();
                            if ui.add_enabled(
                                can_start,
                                egui::Button::new(
                                    egui::RichText::new("▶  Начать зачарование")
                                        .size(17.0)
                                        .strong(),
                                )
                                .min_size(egui::vec2(ui.available_width(), 46.0)),
                            ).clicked() {
                                self.start_automation(ctx);
                            }
                            if !can_start {
                                let hint = if self.pending_capture_id.is_some() {
                                    "Завершается обработка предыдущего кадра. Повторный запуск пока заблокирован."
                                } else if self.selected_window_data().is_none() {
                                    "Сначала выберите игровое окно в шаге 1."
                                } else if !self.inventory_grid_region.is_configured() {
                                    "Сначала настройте сетку инвентаря в шаге 2."
                                } else {
                                    "Дождитесь завершения или остановки текущей серии."
                                };
                                ui.small(egui::RichText::new(hint).color(ui.visuals().weak_text_color()));
                            }

                            ui.horizontal(|ui| {
                                if ui.add_enabled(
                                    self.run_state == RunState::Running,
                                    egui::Button::new("Пауза").min_size(egui::vec2(88.0, 32.0)),
                                ).clicked() {
                                    self.run_state = RunState::Paused;
                                    self.paused_at = Some(Instant::now());
                                    self.status = "Пауза. Для продолжения нажмите «Продолжить»; F12 аварийно остановит серию.".into();
                                    self.log("Автоматизация приостановлена.");
                                }
                                if ui.add_enabled(
                                    self.run_state == RunState::Paused,
                                    egui::Button::new("Продолжить").min_size(egui::vec2(104.0, 32.0)),
                                ).clicked() {
                                    if let Some(paused_at) = self.paused_at.take() {
                                        self.phase_started = self.phase_started + paused_at.elapsed();
                                    }
                                    self.run_state = RunState::Running;
                                    self.next_automation_poll = Instant::now();
                                    self.status = "Автоматизация продолжена.".into();
                                    self.log("Автоматизация продолжена.");
                                }
                                if ui.add_enabled(
                                    self.automation_phase.is_some()
                                        || self.run_state == RunState::Paused,
                                    egui::Button::new("Стоп").min_size(egui::vec2(64.0, 32.0)),
                                ).clicked() {
                                    self.stop_automation("Автоматизация остановлена пользователем.");
                                }
                            });
                        });

                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("ТЕКУЩЕЕ СОСТОЯНИЕ").strong());
                            let (state, color) = match self.run_state {
                                RunState::Running => ("Идёт зачарование", egui::Color32::from_rgb(70, 190, 125)),
                                RunState::Paused => ("Серия приостановлена", egui::Color32::from_rgb(225, 175, 65)),
                                RunState::Stopped => ("Серия завершена или остановлена", ui.visuals().text_color()),
                                RunState::Idle => ("Ожидание запуска", ui.visuals().weak_text_color()),
                            };
                            ui.label(egui::RichText::new(state).strong().color(color));
                            ui.add_space(3.0);
                            ui.label(&self.status);
                            ui.small(&self.summary);
                            ui.separator();

                            egui::Grid::new("run_metrics")
                                .num_columns(2)
                                .spacing(egui::vec2(12.0, 6.0))
                                .show(ui, |ui| {
                                    ui.label("Проход");
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} / {}",
                                            self.current_pass, self.passes_total
                                        ))
                                        .strong()
                                        .monospace(),
                                    );
                                    ui.end_row();

                                    ui.label("Страница");
                                    ui.label(
                                        egui::RichText::new((self.page_index + 1).to_string())
                                            .strong()
                                            .monospace(),
                                    );
                                    ui.end_row();

                                    ui.label("Обработано");
                                    ui.label(
                                        egui::RichText::new(self.items_processed_in_pass.to_string())
                                            .strong()
                                            .monospace(),
                                    );
                                    ui.end_row();

                                    ui.label("Уцелело");
                                    ui.label(
                                        egui::RichText::new(self.surviving_items_in_pass.to_string())
                                            .strong()
                                            .monospace(),
                                    );
                                    ui.end_row();
                                });
                        });

                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label(egui::RichText::new("КАК ВИДИТ ПРОГРАММА").strong());
                            if let Some(texture) = &self.live_frame_preview {
                                let max_width = ui.available_width().max(1.0);
                                let scale = (max_width / self.live_frame_preview_size.x.max(1.0))
                                    .min(220.0 / self.live_frame_preview_size.y.max(1.0))
                                    .min(1.0);
                                ui.add(egui::Image::new((texture.id(), self.live_frame_preview_size * scale)));
                            } else {
                                ui.small("Кадр появится после автоматического поиска инвентаря или запуска серии.");
                            }
                        });

                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(format!("ЖУРНАЛ СОБЫТИЙ · {}", self.logs.len())).strong());
                            egui::ScrollArea::vertical()
                                .max_height(180.0)
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    for line in self.logs.iter().rev().take(120).rev() {
                                        ui.monospace(line);
                                    }
                                });
                        });
                    });
                });
            });
        });

        self.persist_settings_if_needed();
        if self.settings_dirty_since.is_some() {
            let delay = self
                .last_settings_save_attempt
                .map(|last| Duration::from_secs(5).saturating_sub(last.elapsed()))
                .unwrap_or(Duration::from_millis(650))
                .max(Duration::from_millis(50));
            ctx.request_repaint_after(delay);
        }
        if self.automation_phase.is_some()
            || self.pending_capture_id.is_some()
            || !self.window_preview_queue.is_empty()
        {
            ctx.request_repaint_after(Duration::from_millis(25));
        }
    }
}

impl Drop for EnchantmentHelperApp {
    fn drop(&mut self) {
        if let Err(error) = crate::settings::save(&self.current_user_settings()) {
            eprintln!("Не удалось сохранить настройки при закрытии: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn window_preview_size_fits_bounds_and_preserves_aspect_ratio() {
        let size = super::window_preview_size(1920, 1080);
        assert!(size.width <= super::WINDOW_PREVIEW_MAX_WIDTH);
        assert!(size.height <= super::WINDOW_PREVIEW_MAX_HEIGHT);
        assert!((f64::from(size.width) / f64::from(size.height) - 16.0 / 9.0).abs() < 0.02);
    }

    #[test]
    fn small_window_preview_is_not_upscaled() {
        let size = super::window_preview_size(100, 70);
        assert_eq!((size.width, size.height), (100, 70));
    }

    #[test]
    fn detection_streak_requires_the_same_button_location() {
        let first = super::Region { x: 10, y: 20, width: 82, height: 42 };
        let jittered = super::Region { x: 12, y: 19, width: 82, height: 42 };
        let different = super::Region { x: 80, y: 20, width: 82, height: 42 };
        assert!(super::same_detection_region(first, jittered));
        assert!(!super::same_detection_region(first, different));
    }

    #[test]
    fn inventory_page_fingerprint_ignores_small_visual_jitter() {
        let first = 0b1010_1010_u64;
        let jittered = 0b1010_1000_u64;
        let other_page = 0b0101_0101_u64;
        assert!(super::same_inventory_page(first, jittered));
        assert!(!super::same_inventory_page(first, other_page));
    }

    #[test]
    fn one_pass_stops_after_the_first_full_inventory_sweep() {
        assert!(!super::should_start_another_pass(1, 1, 12));
    }

    #[test]
    fn another_full_pass_starts_only_when_surviving_items_remain() {
        assert!(super::should_start_another_pass(1, 2, 12));
        assert!(!super::should_start_another_pass(1, 2, 0));
        assert!(!super::should_start_another_pass(2, 2, 12));
    }
}
