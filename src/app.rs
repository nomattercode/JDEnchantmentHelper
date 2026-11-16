use eframe::egui;
use opencv::prelude::*;
use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use crate::model::{GameWindow, RunState, MAX_PASSES};
use crate::platform::{
    capture_window, click_window_point, f12_pressed, list_visible_windows,
    right_click_window_point, scroll_window,
};
use crate::vision::{
    analyze_enchant_result, analyze_enchant_window, mat_to_rgb, scan_inventory,
    inventory_scanner::{CellDebug, CellKind, InventoryScan},
    EnchantDetectorConfig, InventoryConfig, Region,
};

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
    let bytes = roi
        .data_bytes()
        .map_err(|error| format!("Не удалось получить пиксели страницы: {error}"))?;

    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutomationPhase {
    DetectEnchantDialog,
    InsertAmulet,
    InsertItem,
    WaitingForStart,
    WaitingForOk,
    ReconcileInventory,
    CheckNextPage,
    ReturnToFirstPage,
}

pub struct EnchantmentHelperApp {
    windows: Vec<GameWindow>,
    selected_window: Option<isize>,

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
    return_scroll_remaining: usize,

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
        Self {
            windows: list_visible_windows(),
            selected_window: None,
            calibration_preview: None,
            calibration_preview_size: egui::Vec2::ZERO,
            select_inventory_region: false,
            region_drag_start: None,

            passes_total: 1,
            current_pass: 1,
            items_processed_in_pass: 0,
            surviving_items_in_pass: 0,
            page_index: 0,
            current_page_fingerprint: None,
            unchanged_page_checks: 0,
            return_scroll_remaining: 0,

            run_state: RunState::Idle,
            automation_phase: None,
            consecutive_matches: 0,
            last_detection: None,
            next_automation_poll: Instant::now(),
            last_wait_log: Instant::now(),
            phase_started: Instant::now(),
            paused_at: None,

            inventory_grid_region: Region::default(),
            logical_inventory: Vec::new(),
            processed_slots_current_page: HashSet::new(),
            amulet_slot: None,
            amulet_inserted: false,
            current_item: None,

            status: "Выберите окно игры и настройте сетку инвентаря. Сначала предполагается открытой первая страница.".into(),
            summary: "Ожидание выбора игрового окна.".into(),
            inventory: InventoryConfig::default(),
            enchant: EnchantDetectorConfig::default(),
            logs: vec!["Программа готова. F12 — аварийная остановка.".into()],
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

    fn refresh_windows(&mut self) {
        self.windows = list_visible_windows();
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

        match capture_window(handle).and_then(|frame| {
            let (width, height, pixels) =
                mat_to_rgb(&frame).map_err(|error| error.to_string())?;
            Ok((width, height, pixels))
        }) {
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

    fn begin_scroll_down(&mut self, handle: isize) -> Result<(), String> {
        let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
        let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
        scroll_window(handle, x, y, -120)?;
        self.unchanged_page_checks = 0;
        self.automation_phase = Some(AutomationPhase::CheckNextPage);
        self.phase_started = Instant::now();
        self.next_automation_poll = Instant::now() + Duration::from_millis(750);
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
            self.automation_phase = Some(AutomationPhase::InsertItem);
            self.phase_started = Instant::now();
            self.next_automation_poll = Instant::now() + Duration::from_millis(300);
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

        if self.current_pass >= self.passes_total || self.surviving_items_in_pass == 0 {
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

    fn start_automation(&mut self) {
        if matches!(self.run_state, RunState::Running | RunState::Paused) {
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

        let handle = self.selected_window.expect("checked above");
        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = format!("Не удалось получить начальный кадр игры: {error}");
                return;
            }
        };

        self.current_pass = 1;
        self.items_processed_in_pass = 0;
        self.surviving_items_in_pass = 0;
        self.page_index = 0;
        self.current_page_fingerprint = None;
        self.unchanged_page_checks = 0;
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
        self.next_automation_poll = Instant::now() + Duration::from_millis(250);
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

    fn update_automation(&mut self) {
        if self.automation_phase.is_some() && f12_pressed() {
            self.stop_automation("Аварийная остановка: нажата F12.");
            return;
        }
        if self.run_state != RunState::Running || self.automation_phase.is_none() {
            return;
        }

        let now = Instant::now();
        if now < self.next_automation_poll {
            return;
        }
        self.next_automation_poll = now + Duration::from_millis(260);

        let Some(handle) = self.selected_window else {
            self.stop_automation("Игровое окно не выбрано.");
            return;
        };
        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.stop_automation(&format!("Не удалось получить кадр игры: {error}"));
                return;
            }
        };

        match self.automation_phase {
            Some(AutomationPhase::DetectEnchantDialog) => {
                let analysis = match analyze_enchant_window(&frame, &self.enchant) {
                    Ok(analysis) => analysis,
                    Err(error) => {
                        self.stop_automation(&format!("Ошибка поиска окна зачарования: {error}"));
                        return;
                    }
                };
                if analysis.dialog_window.is_some() {
                    self.log("Окно зачарования найдено. Выбираю амулет.");
                    self.automation_phase = Some(AutomationPhase::InsertAmulet);
                    self.reset_detection_streak();
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(350);
                    self.status = "Окно зачарования найдено. Выбор амулета.".into();
                } else if now.duration_since(self.phase_started) >= Duration::from_secs(30) {
                    self.stop_automation("Окно зачарования не найдено за 30 секунд. Клики не выполнялись.");
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("Ожидание окна зачарования.");
                }
                return;
            }
            Some(AutomationPhase::InsertAmulet) => {
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
                        self.automation_phase = Some(AutomationPhase::InsertItem);
                        self.reset_detection_streak();
                        self.phase_started = Instant::now();
                        self.next_automation_poll = Instant::now() + Duration::from_millis(420);
                        self.status = "Амулет помещён в окно зачарования. Начинаю перебор предметов.".into();
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
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(200);
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
                        self.next_automation_poll = Instant::now() + Duration::from_millis(420);
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
                    self.automation_phase = Some(AutomationPhase::InsertItem);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(300);
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

                if self.current_page_fingerprint == Some(fingerprint) {
                    if self.unchanged_page_checks == 0 {
                        self.unchanged_page_checks = 1;
                        self.next_automation_poll = Instant::now() + Duration::from_millis(500);
                        return;
                    }
                    self.log(format!("Следующая страница после {} не обнаружена.", self.page_index + 1));
                    self.finish_pass_at_bottom(handle, &frame);
                    return;
                }

                self.unchanged_page_checks = 0;
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
                    self.automation_phase = Some(AutomationPhase::InsertItem);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(280);
                } else if let Err(error) = self.begin_scroll_down(handle) {
                    self.stop_automation(&format!("Не удалось перейти к следующей странице: {error}"));
                }
                return;
            }
            Some(AutomationPhase::ReturnToFirstPage) => {
                if self.return_scroll_remaining > 0 {
                    let x = self.inventory_grid_region.x + self.inventory_grid_region.width / 2;
                    let y = self.inventory_grid_region.y + self.inventory_grid_region.height / 2;
                    if let Err(error) = scroll_window(handle, x, y, 120) {
                        self.stop_automation(&format!("Не удалось вернуться к первой странице: {error}"));
                        return;
                    }
                    self.return_scroll_remaining = self.return_scroll_remaining.saturating_sub(1);
                    self.page_index = self.page_index.saturating_sub(1);
                    self.next_automation_poll = Instant::now() + Duration::from_millis(650);
                    self.status = format!(
                        "Возврат к первой странице. Осталось прокруток вверх: {}.",
                        self.return_scroll_remaining
                    );
                    return;
                }

                self.start_next_pass(handle, &frame);
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
            Some(AutomationPhase::ReturnToFirstPage) => Duration::from_secs(30),
            _ => Duration::from_secs(30),
        };
        if now.duration_since(self.phase_started) > timeout {
            let message = match phase {
                Some(AutomationPhase::WaitingForStart) => "Не дождался активации кнопки «Начать».",
                Some(AutomationPhase::WaitingForOk) => "Не дождался появления кнопки OK.",
                Some(AutomationPhase::CheckNextPage) => "Не удалось определить результат прокрутки инвентаря.",
                Some(AutomationPhase::ReturnToFirstPage) => "Не удалось вернуться к первой странице инвентаря.",
                _ => "Истекло время ожидания этапа автоматизации.",
            };
            self.stop_automation(message);
            return;
        }

        let analysis_result = if phase == Some(AutomationPhase::WaitingForOk) {
            analyze_enchant_result(&frame, &self.enchant)
        } else {
            analyze_enchant_window(&frame, &self.enchant)
        };
        let analysis = match analysis_result {
            Ok(analysis) => analysis,
            Err(error) => {
                self.stop_automation(&format!("Ошибка распознавания окна зачарования: {error}"));
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
                            self.next_automation_poll = Instant::now() + Duration::from_millis(550);
                            self.status = format!(
                                "Проход {} из {} · страница {} · зачарование запущено. Ожидание OK.",
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
                            self.log("Кнопка OK подтверждена и нажата. Обновляю текущую страницу инвентаря.");
                            self.reset_detection_streak();
                            self.automation_phase = Some(AutomationPhase::ReconcileInventory);
                            self.phase_started = Instant::now();
                            self.next_automation_poll = Instant::now() + Duration::from_millis(750);
                        }
                        Err(error) => self.stop_automation(&format!("Не удалось нажать OK: {error}")),
                    }
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("Ожидание кнопки OK.");
                }
            }
            _ => {}
        }
    }

    fn logical_inventory_ui(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Проход {} из {} · страница {} · обработано на странице: {}",
            self.current_pass,
            self.passes_total,
            self.page_index + 1,
            self.processed_slots_current_page.len()
        ));
        if self.logical_inventory.is_empty() {
            ui.label("Слоты появятся после начала серии зачарования.");
            return;
        }

        egui::Grid::new("logical_inventory_table")
            .spacing(egui::vec2(8.0, 4.0))
            .show(ui, |ui| {
                for row in 0..self.inventory.rows {
                    for col in 0..self.inventory.columns {
                        let cell = self
                            .logical_inventory
                            .iter()
                            .find(|cell| cell.row == row && cell.col == col);

                        let (symbol, color, hint) = match cell {
                            Some(cell) if cell.kind == CellKind::Amulet => {
                                ("А", egui::Color32::YELLOW, "Ячейка амулета")
                            }
                            Some(cell)
                                if cell.kind == CellKind::Occupied
                                    && self.processed_slots_current_page.contains(&(row, col)) =>
                            {
                                ("✓", egui::Color32::LIGHT_GRAY, "Уже обработан в этом проходе")
                            }
                            Some(cell) if cell.kind == CellKind::Occupied => {
                                ("●", egui::Color32::LIGHT_GREEN, "Предмет ожидает зачарования")
                            }
                            Some(cell) if cell.kind == CellKind::BeforeStart => {
                                ("—", egui::Color32::GRAY, "Слот до начальной позиции")
                            }
                            Some(cell) if cell.kind == CellKind::OutOfFrame => {
                                ("?", egui::Color32::GRAY, "За границами кадра")
                            }
                            Some(_) => ("·", egui::Color32::LIGHT_RED, "Пустая ячейка"),
                            None => ("?", egui::Color32::GRAY, "Ячейка не просканирована"),
                        };
                        ui.colored_label(color, egui::RichText::new(symbol).strong().monospace())
                            .on_hover_text(format!("Строка {}, столбец {}: {hint}", row + 1, col + 1));
                    }
                    ui.end_row();
                }
            });
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

        ui.add(
            egui::Slider::new(&mut self.inventory.threshold, 0.0..=1.0)
                .step_by(0.01)
                .text("Чувствительность определения предмета"),
        );

        if self.inventory_grid_region.is_configured() {
            ui.label(format!(
                "Сетка настроена: {}×{} px, шаг {}×{} px.",
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

    fn game_window_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label("Игровое окно");
            let selection_enabled = self.automation_phase.is_none();
            let previous_window = self.selected_window;
            ui.add_enabled_ui(selection_enabled, |ui| {
                egui::ComboBox::from_id_salt("game_window")
                    .width((ui.available_width() - 90.0).max(180.0))
                    .selected_text(self.selected_label())
                    .show_ui(ui, |ui| {
                        for window in &self.windows {
                            let label = format!(
                                "{} · {}×{}",
                                window.title, window.width, window.height
                            );
                            ui.selectable_value(
                                &mut self.selected_window,
                                Some(window.handle),
                                label,
                            );
                        }
                    });
                if ui.button("↻").on_hover_text("Обновить список окон").clicked() {
                    self.refresh_windows();
                }
            });

            if self.selected_window != previous_window {
                self.calibration_preview = None;
                self.select_inventory_region = false;
                self.inventory_grid_region = Region::default();
                self.logical_inventory.clear();
                self.processed_slots_current_page.clear();
                self.amulet_slot = None;
                self.amulet_inserted = false;
                self.current_item = None;
                self.page_index = 0;
                self.current_page_fingerprint = None;
                self.status = "Игровое окно выбрано. Первая страница инвентаря должна быть открыта. Снимите кадр для настройки сетки.".into();
                self.log("Выбрано новое игровое окно. Настройки сетки сброшены.");
            }
        });

        if self.selected_window_data().is_none() {
            ui.label("Выберите открытое окно игры.");
            return;
        }

        ui.horizontal(|ui| {
            ui.add_enabled_ui(
                self.automation_phase.is_none(),
                |ui| {
                    if ui.button("Снять кадр и настроить сетку").clicked() {
                        self.capture_calibration_snapshot(ctx);
                    }
                },
            );
            if self.calibration_preview.is_some() {
                ui.label("Перетащите мышь по всей сетке инвентаря.");
            } else if self.inventory_grid_region.is_configured() {
                ui.label("Сетка настроена. Превью отключено до следующей перенастройки.");
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
        self.update_automation();

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Помощник зачарования Jade Dynasty");
                ui.separator();
                let state = match self.run_state {
                    RunState::Running => "РАБОТАЕТ",
                    RunState::Paused => "ПАУЗА",
                    RunState::Stopped => "ОСТАНОВЛЕНО",
                    RunState::Idle => "ГОТОВО",
                };
                ui.label(state);
                ui.separator();
                ui.colored_label(egui::Color32::LIGHT_RED, "F12 — АВАРИЙНЫЙ СТОП");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.group(|ui| self.game_window_ui(ui, ctx));

                ui.add_space(8.0);
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Полных проходов").strong());
                        if ui.add_enabled(
                            self.run_state != RunState::Running
                                && self.run_state != RunState::Paused
                                && self.passes_total > 1,
                            egui::Button::new("◀"),
                        ).clicked() {
                            self.passes_total = self.passes_total.saturating_sub(1).max(1);
                        }
                        ui.label(
                            egui::RichText::new(self.passes_total.to_string())
                                .size(22.0)
                                .strong()
                                .monospace(),
                        );
                        if ui.add_enabled(
                            self.run_state != RunState::Running
                                && self.run_state != RunState::Paused
                                && self.passes_total < MAX_PASSES,
                            egui::Button::new("▶"),
                        ).clicked() {
                            self.passes_total = self.passes_total.saturating_add(1).min(MAX_PASSES);
                        }
                    });

                    ui.horizontal_wrapped(|ui| {
                        if ui.add_enabled(
                            self.run_state != RunState::Running
                                && self.run_state != RunState::Paused
                                && self.selected_window.is_some()
                                && self.inventory_grid_region.is_configured(),
                            egui::Button::new("Начать"),
                        ).clicked() {
                            self.start_automation();
                        }
                        if ui.add_enabled(
                            self.run_state == RunState::Running,
                            egui::Button::new("Пауза"),
                        ).clicked() {
                            self.run_state = RunState::Paused;
                            self.paused_at = Some(Instant::now());
                            self.status = "Пауза. F12 продолжает работать.".into();
                            self.log("Автоматизация приостановлена.");
                        }
                        if ui.add_enabled(
                            self.run_state == RunState::Paused,
                            egui::Button::new("Продолжить"),
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
                            egui::Button::new("Стоп"),
                        ).clicked() {
                            self.stop_automation("Автоматизация остановлена пользователем.");
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Проход {} из {}",
                            self.current_pass, self.passes_total
                        ));
                        ui.separator();
                        ui.label(format!("Страница {}", self.page_index + 1));
                        ui.separator();
                        ui.label(format!(
                            "Обработано: {} · уцелело: {}",
                            self.items_processed_in_pass, self.surviving_items_in_pass
                        ));
                    });
                });

                ui.group(|ui| {
                    ui.label(egui::RichText::new(&self.status).strong());
                    ui.label(&self.summary);
                });

                ui.collapsing("Настройки инвентаря", |ui| {
                    ui.add_enabled_ui(self.automation_phase.is_none(), |ui| {
                        self.inventory_settings_ui(ui);
                    });
                    if self.automation_phase.is_some() {
                        ui.small("Настройки заблокированы на время серии.");
                    }
                });

                ui.collapsing("Таблица текущей страницы", |ui| {
                    self.logical_inventory_ui(ui);
                });

                ui.heading("Журнал");
                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in self.logs.iter().rev().take(120).rev() {
                            ui.monospace(line);
                        }
                    });
            });
        });

        if self.automation_phase.is_some() {
            ctx.request_repaint_after(Duration::from_millis(35));
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn detection_streak_requires_the_same_button_location() {
        let first = super::Region { x: 10, y: 20, width: 82, height: 42 };
        let jittered = super::Region { x: 12, y: 19, width: 82, height: 42 };
        let different = super::Region { x: 80, y: 20, width: 82, height: 42 };
        assert!(super::same_detection_region(first, jittered));
        assert!(!super::same_detection_region(first, different));
    }
}
