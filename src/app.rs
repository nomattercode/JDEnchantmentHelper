use eframe::egui;
use opencv::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use crate::model::{GameWindow, RunState, MAX_RUNS};
use crate::platform::{
    capture_window, click_window_point, f12_pressed, list_visible_windows,
    right_click_window_point,
};
use crate::vision::{
    analyze_enchant_window, load_image, mat_to_rgb, save_png, scan_inventory,
    inventory_scanner::{CellDebug, CellKind, InventoryScan},
    EnchantDetectorConfig, InventoryConfig, Region,
};

fn validate_inventory_scan(
    scan: &InventoryScan,
    region: Region,
    frame: &opencv::core::Mat,
) -> Result<(), String> {
    if !region.is_configured() || region.x < 0 || region.y < 0 {
        return Err("Область сетки инвентаря не откалибрована.".into());
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
        return Err(format!(
            "Откалиброванная область инвентаря ({},{} {}x{}) выходит за кадр {}x{}.",
            region.x,
            region.y,
            region.width,
            region.height,
            frame.cols(),
            frame.rows()
        ));
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
                "Ячейка r{}c{} ({},{},{}x{}) выходит за откалиброванную сетку. Перевыделите сетку или проверьте геометрию.",
                cell.row + 1,
                cell.col + 1,
                cell.rect.x,
                cell.rect.y,
                cell.rect.width,
                cell.rect.height
            ));
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Inventory,
    Enchant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutomationPhase {
    DetectEnchantDialog,
    InsertAmulet,
    InsertItem,
    WaitingForStart,
    WaitingForOk,
    ReconcileInventory,
}

pub struct EnchantmentHelperApp {
    windows: Vec<GameWindow>,
    selected_window: Option<isize>,
    window_previews: HashMap<isize, egui::TextureHandle>,
    window_preview_errors: HashMap<isize, String>,
    next_window_refresh: Instant,
    runs: u8,
    completed_runs: u8,
    run_state: RunState,
    automation_phase: Option<AutomationPhase>,
    consecutive_matches: u8,
    next_automation_poll: Instant,
    last_wait_log: Instant,
    phase_started: Instant,
    paused_at: Option<Instant>,
    inventory_grid_region: Region,
    logical_inventory: Vec<CellDebug>,
    attempted_items: HashSet<(u8, u8)>,
    amulet_slot: Option<CellDebug>,
    current_item: Option<CellDebug>,
    last_start_button: Option<Region>,
    last_enchant_window: Option<Region>,
    status: String,
    image_path: String,
    output_dir: String,
    mode: Mode,
    inventory: InventoryConfig,
    enchant: EnchantDetectorConfig,
    save_debug: bool,
    preview: Option<egui::TextureHandle>,
    preview_size: egui::Vec2,
    summary: String,
    logs: Vec<String>,
    select_start_region: bool,
    select_inventory_region: bool,
    region_drag_start: Option<egui::Pos2>,
}

impl Default for EnchantmentHelperApp {
    fn default() -> Self {
        Self {
            windows: list_visible_windows(),
            selected_window: None,
            window_previews: HashMap::new(),
            window_preview_errors: HashMap::new(),
            next_window_refresh: Instant::now(),
            runs: 1,
            completed_runs: 0,
            run_state: RunState::Idle,
            automation_phase: None,
            consecutive_matches: 0,
            next_automation_poll: Instant::now(),
            last_wait_log: Instant::now(),
            phase_started: Instant::now(),
            paused_at: None,
            inventory_grid_region: Region::default(),
            logical_inventory: Vec::new(),
            attempted_items: HashSet::new(),
            amulet_slot: None,
            current_item: None,
            last_start_button: None,
            last_enchant_window: None,
            status: "Выберите окно игры и один раз выделите область сетки инвентаря.".into(),
            image_path: String::new(),
            output_dir: "diagnostics".into(),
            mode: Mode::Enchant,
            inventory: InventoryConfig::default(),
            enchant: EnchantDetectorConfig::default(),
            save_debug: true,
            preview: None,
            preview_size: egui::Vec2::ZERO,
            summary: "Ожидание игрового окна.".into(),
            logs: vec!["[app] Ready. F12 is the global emergency stop.".into()],
            select_start_region: false,
            select_inventory_region: false,
            region_drag_start: None,
        }
    }
}

impl EnchantmentHelperApp {
    fn log(&mut self, line: impl Into<String>) {
        self.logs.push(line.into());
        if self.logs.len() > 400 {
            self.logs.drain(0..self.logs.len() - 400);
        }
    }

    fn selected_window_data(&self) -> Option<&GameWindow> {
        self.selected_window
            .and_then(|handle| self.windows.iter().find(|window| window.handle == handle))
    }

    fn selected_label(&self) -> String {
        self.selected_window_data()
            .map(|window| {
                format!(
                    "{} · PID {} · HWND 0x{:X} · {}×{}",
                    window.title, window.process_id, window.handle, window.width, window.height
                )
            })
            .unwrap_or_else(|| "Выберите окно игры…".into())
    }

    fn refresh_windows(&mut self, ctx: &egui::Context) {
        self.windows = list_visible_windows();
        let valid: std::collections::HashSet<isize> =
            self.windows.iter().map(|window| window.handle).collect();
        self.window_previews.retain(|handle, _| valid.contains(handle));
        self.window_preview_errors.retain(|handle, _| valid.contains(handle));

        if self
            .selected_window
            .is_some_and(|handle| !valid.contains(&handle))
        {
            self.selected_window = None;
            if self.automation_phase.is_some() {
                self.stop_automation("Выбранное игровое окно исчезло из списка.");
            }
        }

        self.capture_selected_preview(ctx);
        self.next_window_refresh = Instant::now() + Duration::from_secs(2);
        self.log(format!(
            "[window] refreshed; visible windows={}",
            self.windows.len()
        ));
    }

    fn capture_selected_preview(&mut self, ctx: &egui::Context) {
        let Some(handle) = self.selected_window else {
            return;
        };

        match capture_window(handle).and_then(|mat| {
            let (width, height, pixels) = mat_to_rgb(&mat).map_err(|error| error.to_string())?;
            Ok((mat, width, height, pixels))
        }) {
            Ok((_mat, width, height, pixels)) => {
                let image = egui::ColorImage::from_rgb([width, height], &pixels);
                let texture = ctx.load_texture(
                    format!("window-live-{handle}"),
                    image,
                    egui::TextureOptions::LINEAR,
                );
                self.window_previews.insert(handle, texture);
                self.window_preview_errors.remove(&handle);
            }
            Err(error) => {
                self.window_preview_errors.insert(handle, error.clone());
                self.log(format!("[window][0x{handle:X}] {error}"));
            }
        }
    }

    fn set_preview(&mut self, ctx: &egui::Context, mat: &opencv::core::Mat) {
        match mat_to_rgb(mat) {
            Ok((width, height, pixels)) => {
                let image = egui::ColorImage::from_rgb([width, height], &pixels);
                self.preview_size = egui::vec2(width as f32, height as f32);
                if let Some(texture) = &mut self.preview {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.preview = Some(ctx.load_texture(
                        "vision-preview",
                        image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
            Err(error) => self.log(format!("[preview] {error}")),
        }
    }

    fn save_debug(&mut self, name: &str, mat: &opencv::core::Mat) {
        if !self.save_debug {
            return;
        }
        let path = std::path::Path::new(&self.output_dir).join(name);
        match save_png(&path, mat) {
            Ok(()) => self.log(format!("[debug] saved {}", path.display())),
            Err(error) => self.log(format!("[debug] {error}")),
        }
    }

    fn analyze_file(&mut self, ctx: &egui::Context) {
        if self.image_path.trim().is_empty() {
            self.status = "Укажите путь к тестовому изображению.".into();
            return;
        }
        self.log(format!("[app] loading {}", self.image_path));
        let frame = match load_image(self.image_path.trim()) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = error.clone();
                self.log(format!("[error] {error}"));
                return;
            }
        };
        self.log(format!(
            "[app] test image={}x{}",
            frame.cols(),
            frame.rows()
        ));
        self.save_debug("input_frame.png", &frame);

        match self.mode {
            Mode::Inventory => match scan_inventory(&frame, &self.inventory) {
                Ok(result) => {
                    self.summary = format!(
                        "Ячеек: {} · кандидатов: {} · метод: {}",
                        result.cells.len(),
                        result.candidates.len(),
                        self.inventory.method.label()
                    );
                    for line in &result.log {
                        self.log(line.clone());
                    }
                    self.save_debug("inventory_overlay.png", &result.overlay);
                    self.save_debug("inventory_score_map.png", &result.score_map);
                    self.set_preview(ctx, &result.overlay);
                    self.status = "Диагностический анализ завершён; игровые действия не выполнялись.".into();
                }
                Err(error) => {
                    self.status = error.to_string();
                    self.log(format!("[error][inventory] {error}"));
                }
            },
            Mode::Enchant => match analyze_enchant_window(&frame, &self.enchant) {
                Ok(result) => {
                    let dialog_text = result
                        .dialog_window
                        .as_ref()
                        .map(|rect| format!("найдено ({},{}) {}×{}", rect.x, rect.y, rect.width, rect.height))
                        .unwrap_or_else(|| "не найдено".into());
                    self.summary = format!(
                        "Окно зачарования: {} · OK: {} · score {:.3} · {} · scale {:.2} · Start: {}",
                        dialog_text,
                        if result.ok_button.found { "найдена" } else { "не найдена" },
                        result.ok_button.score,
                        result.ok_button.template_name,
                        result.ok_button.scale,
                        result.start_button.as_ref().map_or("не найдена", |button| {
                            if button.active { "синяя/активна" } else { "серая/неактивна" }
                        })
                    );
                    for line in &result.log {
                        self.log(line.clone());
                    }
                    self.save_debug("enchant_overlay.png", &result.overlay);
                    self.set_preview(ctx, &result.overlay);
                    self.status = "Диагностический анализ завершён; игровые действия не выполнялись.".into();
                }
                Err(error) => {
                    self.status = error.to_string();
                    self.log(format!("[error][enchant] {error}"));
                }
            },
        }
    }

    fn scan_inventory_snapshot(
        &mut self,
        ctx: &egui::Context,
        frame: &opencv::core::Mat,
        initial: bool,
    ) -> Result<(), String> {
        let scan = scan_inventory(frame, &self.inventory)
            .map_err(|error| format!("Не удалось просканировать инвентарь: {error}"))?;
        validate_inventory_scan(&scan, self.inventory_grid_region, frame)?;

        let amulet = scan
            .cells
            .iter()
            .find(|cell| {
                cell.row == self.inventory.amulet_row
                    && cell.col == self.inventory.amulet_col
            })
            .cloned()
            .ok_or_else(|| "Ячейка амулета не входит в область инвентаря.".to_string())?;

        if amulet.score < self.inventory.threshold {
            return Err(format!(
                "Амулет не распознан в ячейке {}:{} (оценка {:.3}, порог {:.3}).",
                amulet.row + 1,
                amulet.col + 1,
                amulet.score,
                self.inventory.threshold
            ));
        }

        let previous = self.logical_inventory.clone();
        for old in &previous {
            if old.kind != CellKind::Occupied {
                continue;
            }
            if let Some(updated) = scan
                .cells
                .iter()
                .find(|cell| cell.row == old.row && cell.col == old.col)
            {
                if updated.kind == CellKind::Empty {
                    self.log(format!(
                        "[inventory][table] slot r{}c{} changed ITEM -> EMPTY; it will be skipped",
                        old.row + 1,
                        old.col + 1
                    ));
                } else if updated.kind == CellKind::OutOfFrame {
                    self.log(format!(
                        "[inventory][table] slot r{}c{} is outside the captured frame; it will not be clicked",
                        old.row + 1,
                        old.col + 1
                    ));
                }
            }
        }

        self.logical_inventory = scan.cells.clone();
        self.amulet_slot = Some(amulet);
        for line in &scan.log {
            self.log(line.clone());
        }

        let candidate_count = self
            .logical_inventory
            .iter()
            .filter(|cell| cell.kind == CellKind::Occupied)
            .count();
        self.summary = format!(
            "Логическая таблица: {} ячеек · предметов: {} · попыток: {}",
            self.logical_inventory.len(),
            candidate_count,
            self.completed_runs
        );

        self.save_debug("inventory_live_overlay.png", &scan.overlay);
        self.save_debug("inventory_live_score_map.png", &scan.score_map);
        self.set_preview(ctx, &scan.overlay);

        if initial {
            self.log(format!(
                "[inventory][table] initialized: cells={} items={} amulet=r{}c{}",
                self.logical_inventory.len(),
                candidate_count,
                self.inventory.amulet_row + 1,
                self.inventory.amulet_col + 1
            ));
        } else {
            self.log(format!(
                "[inventory][table] refreshed after OK: items currently detected={candidate_count}"
            ));
        }
        Ok(())
    }

    fn scan_inventory_manually(&mut self, ctx: &egui::Context) {
        if self.automation_phase.is_some() {
            self.status = "Нельзя вручную сканировать инвентарь во время автоматизации.".into();
            return;
        }
        let Some(handle) = self.selected_window else {
            self.status = "Сначала выберите игровое окно.".into();
            return;
        };
        if !self.inventory_grid_region.is_configured() {
            self.select_inventory_region = true;
            self.status = "Сначала выделите всю сетку инвентаря на превью.".into();
            return;
        }

        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = format!("Не удалось получить кадр игры: {error}");
                return;
            }
        };

        self.attempted_items.clear();
        self.current_item = None;
        self.completed_runs = 0;
        self.logical_inventory.clear();

        match self.scan_inventory_snapshot(ctx, &frame, true) {
            Ok(()) => {
                self.status = "Сканирование завершено без игровых кликов. Проверьте логическую таблицу перед запуском.".into();
                self.log("[inventory] manual live scan complete; no game input was sent");
            }
            Err(error) => {
                self.status = error.clone();
                self.log(format!("[inventory][error] {error}"));
            }
        }
    }

    fn next_unattempted_item(&self) -> Option<CellDebug> {
        self.logical_inventory
            .iter()
            .find(|cell| {
                cell.kind == CellKind::Occupied
                    && !self.attempted_items.contains(&(cell.row, cell.col))
            })
            .cloned()
    }

    fn start_automation(&mut self, ctx: &egui::Context) {
        if self.run_state == RunState::Running || self.run_state == RunState::Paused {
            return;
        }
        if self.selected_window.is_none() {
            self.status = "Сначала выберите окно игры.".into();
            return;
        }
        if !self.inventory_grid_region.is_configured() {
            self.select_inventory_region = true;
            self.select_start_region = false;
            self.mode = Mode::Inventory;
            self.status = "Перед запуском обведите на превью всю сетку инвентаря — от первой до последней ячейки.".into();
            return;
        }

        let Some(handle) = self.selected_window else {
            return;
        };
        let frame = match capture_window(handle) {
            Ok(frame) => frame,
            Err(error) => {
                self.status = format!("Не удалось получить начальный кадр игры: {error}");
                return;
            }
        };

        self.completed_runs = 0;
        self.paused_at = None;
        self.attempted_items.clear();
        self.current_item = None;
        self.amulet_slot = None;
        self.logical_inventory.clear();

        if let Err(error) = self.scan_inventory_snapshot(ctx, &frame, true) {
            self.run_state = RunState::Stopped;
            self.automation_phase = None;
            self.status = error.clone();
            self.log(format!("[automation] initial inventory scan failed: {error}"));
            return;
        }

        let Some(item) = self.next_unattempted_item() else {
            self.run_state = RunState::Stopped;
            self.automation_phase = None;
            self.status = "В инвентаре не найдено предметов для зачарования после стартовой ячейки.".into();
            return;
        };

        self.current_item = Some(item.clone());
        self.consecutive_matches = 0;
        self.automation_phase = Some(AutomationPhase::DetectEnchantDialog);
        self.run_state = RunState::Running;
        self.phase_started = Instant::now();
        self.next_automation_poll = Instant::now() + Duration::from_millis(250);
        self.last_wait_log = Instant::now();
        self.status = format!(
            "Инвентарь просканирован. Поиск окна зачарования перед выбором амулета и предмета r{}c{}. F12 — аварийный стоп.",
            item.row + 1,
            item.col + 1
        );
        self.log(format!(
            "[automation] armed; target runs={} inventory items={} first=r{}c{}",
            self.runs,
            self.logical_inventory
                .iter()
                .filter(|cell| cell.kind == CellKind::Occupied)
                .count(),
            item.row + 1,
            item.col + 1
        ));
    }

    fn stop_automation(&mut self, reason: &str) {
        let was_active = self.automation_phase.is_some()
            || matches!(self.run_state, RunState::Running | RunState::Paused);
        self.automation_phase = None;
        self.paused_at = None;
        self.consecutive_matches = 0;
        self.run_state = RunState::Stopped;
        self.status = reason.into();
        if was_active {
            self.log(format!("[automation] STOP: {reason}"));
        }
    }

    fn update_automation(&mut self, ctx: &egui::Context) {
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
            self.stop_automation("Окно игры не выбрано. Автоматизация остановлена.");
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
                self.set_preview(ctx, &analysis.overlay);
                if let Some(rect) = analysis.dialog_window {
                    let detected = Region {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    };
                    self.last_enchant_window = Some(detected);
                    let path = std::path::Path::new(&self.output_dir)
                        .join("enchant_window_last_seen.txt");
                    let saved = (|| -> Result<(), String> {
                        if let Some(parent) = path.parent() {
                            std::fs::create_dir_all(parent)
                                .map_err(|error| format!("Не удалось создать папку координат: {error}"))?;
                        }
                        std::fs::write(
                            &path,
                            format!(
                                "window=0x{handle:X}\nx={}\ny={}\nwidth={}\nheight={}\n",
                                detected.x, detected.y, detected.width, detected.height
                            ),
                        )
                        .map_err(|error| format!("Не удалось сохранить координаты окна зачарования: {error}"))
                    })();
                    if let Err(error) = saved {
                        self.log(format!("[automation] {error}"));
                    }
                    self.log(format!(
                        "[automation] enchant dialog detected at ({},{},{},{})",
                        rect.x, rect.y, rect.width, rect.height
                    ));
                    self.automation_phase = Some(AutomationPhase::InsertAmulet);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(350);
                    self.status = "Окно зачарования найдено. Выбор амулета.".into();
                } else if now.duration_since(self.phase_started) >= Duration::from_secs(30) {
                    self.stop_automation("Окно зачарования не найдено за 30 секунд. Клики по инвентарю не выполнялись.");
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("[automation] waiting for enchant dialog header");
                }
                return;
            }
            Some(AutomationPhase::InsertAmulet) => {
                let Some(amulet) = self.amulet_slot.clone() else {
                    self.stop_automation("В логической таблице не задана ячейка амулета.");
                    return;
                };
                let scan = match scan_inventory(&frame, &self.inventory) {
                    Ok(scan) => scan,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось перепроверить ячейку амулета: {error}"));
                        return;
                    }
                };
                if let Err(error) = validate_inventory_scan(&scan, self.inventory_grid_region, &frame) {
                    self.stop_automation(&format!("Проверка границ инвентаря перед ПКМ: {error}"));
                    return;
                }
                let verified = scan
                    .cells
                    .iter()
                    .find(|cell| cell.row == amulet.row && cell.col == amulet.col)
                    .cloned();
                let Some(verified) = verified else {
                    self.stop_automation("Ячейка амулета отсутствует в повторном сканировании.");
                    return;
                };
                if verified.score < self.inventory.threshold || verified.kind == CellKind::OutOfFrame {
                    let message = format!(
                        "Амулет не подтверждён перед ПКМ: r{}c{} score={:.3}, threshold={:.3}.",
                        verified.row + 1,
                        verified.col + 1,
                        verified.score,
                        self.inventory.threshold
                    );
                    self.stop_automation(&message);
                    return;
                }
                self.amulet_slot = Some(verified.clone());
                let x = verified.rect.x + verified.rect.width / 2;
                let y = verified.rect.y + verified.rect.height / 2;
                match right_click_window_point(handle, x, y) {
                    Ok(()) => {
                        self.log(format!(
                            "[automation] verified and right-clicked amulet r{}c{} score={:.3} at ({x},{y})",
                            verified.row + 1,
                            verified.col + 1,
                            verified.score
                        ));
                        self.automation_phase = Some(AutomationPhase::InsertItem);
                        self.phase_started = Instant::now();
                        self.next_automation_poll = Instant::now() + Duration::from_millis(420);
                        self.status = "Амулет перепроверен и выбран. Перенос предмета в окно зачарования.".into();
                    }
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось выбрать амулет: {error}"));
                    }
                }
                return;
            }
            Some(AutomationPhase::InsertItem) => {
                let Some(item) = self.current_item.clone() else {
                    self.stop_automation("Текущий предмет не задан.");
                    return;
                };
                let scan = match scan_inventory(&frame, &self.inventory) {
                    Ok(scan) => scan,
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось перепроверить ячейку предмета: {error}"));
                        return;
                    }
                };
                if let Err(error) = validate_inventory_scan(&scan, self.inventory_grid_region, &frame) {
                    self.stop_automation(&format!("Проверка границ инвентаря перед ПКМ: {error}"));
                    return;
                }
                let verified = scan
                    .cells
                    .iter()
                    .find(|cell| cell.row == item.row && cell.col == item.col)
                    .cloned();
                let Some(verified) = verified else {
                    self.stop_automation("Ячейка выбранного предмета отсутствует в повторном сканировании.");
                    return;
                };

                if verified.kind != CellKind::Occupied || verified.score < self.inventory.threshold {
                    self.logical_inventory = scan.cells.clone();
                    for line in &scan.log {
                        self.log(line.clone());
                    }
                    self.save_debug("inventory_item_missing_before_click.png", &scan.overlay);
                    self.set_preview(ctx, &scan.overlay);
                    self.log(format!(
                        "[automation] skipped r{}c{} before right-click: slot is {:?}, score={:.3}",
                        item.row + 1,
                        item.col + 1,
                        verified.kind,
                        verified.score
                    ));
                    self.automation_phase = Some(AutomationPhase::ReconcileInventory);
                    self.phase_started = Instant::now();
                    self.next_automation_poll = Instant::now() + Duration::from_millis(350);
                    self.status = "Выбранная ячейка стала пустой до клика. Таблица обновляется; предмет будет пропущен.".into();
                    return;
                }

                self.logical_inventory = scan.cells.clone();
                self.current_item = Some(verified.clone());
                let x = verified.rect.x + verified.rect.width / 2;
                let y = verified.rect.y + verified.rect.height / 2;
                match right_click_window_point(handle, x, y) {
                    Ok(()) => {
                        self.log(format!(
                            "[automation] verified and right-clicked item r{}c{} score={:.3} at ({x},{y})",
                            verified.row + 1,
                            verified.col + 1,
                            verified.score
                        ));
                        self.automation_phase = Some(AutomationPhase::WaitingForStart);
                        self.phase_started = Instant::now();
                        self.consecutive_matches = 0;
                        self.next_automation_poll = Instant::now() + Duration::from_millis(420);
                        self.status = "Предмет перепроверен и выбран. Ожидание появления и активации кнопки «Начать».".into();
                    }
                    Err(error) => {
                        self.stop_automation(&format!("Не удалось выбрать предмет: {error}"));
                    }
                }
                return;
            }
            Some(AutomationPhase::ReconcileInventory) => {
                let current_item = self.current_item.clone();
                if let Err(error) = self.scan_inventory_snapshot(ctx, &frame, false) {
                    self.stop_automation(&format!(
                        "Не удалось обновить таблицу инвентаря после OK: {error}"
                    ));
                    return;
                }

                if let Some(previous) = current_item {
                    let updated_kind = self
                        .logical_inventory
                        .iter()
                        .find(|cell| cell.row == previous.row && cell.col == previous.col)
                        .map(|cell| cell.kind);
                    match updated_kind {
                        Some(CellKind::Empty) => {
                            self.log(format!(
                                "[automation] current item r{}c{} is now empty (possibly destroyed); removed from the available queue",
                                previous.row + 1,
                                previous.col + 1
                            ));
                        }
                        Some(CellKind::Occupied) => {
                            self.log(format!(
                                "[automation] current item r{}c{} remains in inventory; it is marked attempted and will not be clicked again",
                                previous.row + 1,
                                previous.col + 1
                            ));
                        }
                        _ => {}
                    }
                }

                if self.completed_runs >= self.runs {
                    self.automation_phase = None;
                    self.current_item = None;
                    self.run_state = RunState::Stopped;
                    self.status = format!(
                        "Серия завершена: выполнено {} из {} прогонов. Инвентарь повторно просканирован.",
                        self.completed_runs, self.runs
                    );
                    self.log("[automation] requested run count reached after inventory reconciliation");
                    return;
                }

                let Some(next_item) = self.next_unattempted_item() else {
                    self.automation_phase = None;
                    self.current_item = None;
                    self.run_state = RunState::Stopped;
                    self.status = format!(
                        "Остановлено: доступные предметы закончились. Выполнено {} из {} прогонов.",
                        self.completed_runs, self.runs
                    );
                    self.log("[automation] no unattempted occupied inventory slots remain");
                    return;
                };

                self.current_item = Some(next_item.clone());
                self.automation_phase = Some(AutomationPhase::DetectEnchantDialog);
                self.phase_started = Instant::now();
                self.next_automation_poll = Instant::now() + Duration::from_millis(450);
                self.consecutive_matches = 0;
                self.status = format!(
                    "Таблица обновлена. Следующий предмет: r{}c{} (прогон {} из {}).",
                    next_item.row + 1,
                    next_item.col + 1,
                    self.completed_runs + 1,
                    self.runs
                );
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
            _ => Duration::from_secs(30),
        };
        if now.duration_since(self.phase_started) > timeout {
            let message = match phase {
                Some(AutomationPhase::DetectEnchantDialog) => {
                    "Тайм-аут: окно зачарования не обнаружено."
                }
                Some(AutomationPhase::WaitingForStart) => {
                    "Тайм-аут: кнопка «Начать» не появилась или осталась неактивной."
                }
                Some(AutomationPhase::WaitingForOk) => {
                    "Тайм-аут: окно не показало распознаваемую кнопку OK."
                }
                _ => "Тайм-аут этапа автоматизации.",
            };
            self.stop_automation(message);
            return;
        }

        let analysis = match analyze_enchant_window(&frame, &self.enchant) {
            Ok(analysis) => analysis,
            Err(error) => {
                self.stop_automation(&format!("Ошибка анализа OpenCV: {error}"));
                return;
            }
        };
        self.set_preview(ctx, &analysis.overlay);

        if let Some(dialog) = analysis.dialog_window {
            let detected_dialog = Region {
                x: dialog.x,
                y: dialog.y,
                width: dialog.width,
                height: dialog.height,
            };
            if self.last_enchant_window != Some(detected_dialog) {
                self.last_enchant_window = Some(detected_dialog);
                let path = std::path::Path::new(&self.output_dir)
                    .join("enchant_window_last_seen.txt");
                let saved = (|| -> Result<(), String> {
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|error| format!("Не удалось создать папку координат: {error}"))?;
                    }
                    std::fs::write(
                        &path,
                        format!(
                            "window=0x{handle:X}\nx={}\ny={}\nwidth={}\nheight={}\n",
                            detected_dialog.x, detected_dialog.y,
                            detected_dialog.width, detected_dialog.height
                        ),
                    )
                    .map_err(|error| format!("Не удалось сохранить координаты окна зачарования: {error}"))
                })();
                match saved {
                    Ok(()) => self.log(format!(
                        "[automation] saved enchant-window coordinates to {}",
                        path.display()
                    )),
                    Err(error) => self.log(format!("[automation] {error}")),
                }
            }
        }

        if let Some(button) = &analysis.start_button {
            let detected = Region {
                x: button.rect.x,
                y: button.rect.y,
                width: button.rect.width,
                height: button.rect.height,
            };
            if self.last_start_button != Some(detected) {
                self.last_start_button = Some(detected);
                let path = std::path::Path::new(&self.output_dir)
                    .join("start_button_last_seen.txt");
                let saved = (|| -> Result<(), String> {
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)
                            .map_err(|error| format!("Не удалось создать папку координат: {error}"))?;
                    }
                    std::fs::write(
                        &path,
                        format!(
                            "window=0x{handle:X}\nx={}\ny={}\nwidth={}\nheight={}\n",
                            detected.x, detected.y, detected.width, detected.height
                        ),
                    )
                    .map_err(|error| format!("Не удалось сохранить координаты START: {error}"))
                })();
                match saved {
                    Ok(()) => self.log(format!(
                        "[automation] saved START coordinates to {}",
                        path.display()
                    )),
                    Err(error) => self.log(format!("[automation] {error}")),
                }
            }
            self.summary = format!(
                "START: {} · синий {:.0}% · OK {:.3} · предмет r{}c{} · прогон {}/{}",
                if button.active { "активна" } else { "неактивна" },
                button.blue_ratio * 100.0,
                analysis.ok_button.score,
                self.current_item.as_ref().map_or(0, |item| item.row + 1),
                self.current_item.as_ref().map_or(0, |item| item.col + 1),
                self.completed_runs.saturating_add(1).min(self.runs),
                self.runs
            );
            if self.last_wait_log.elapsed() >= Duration::from_secs(10) {
                self.last_wait_log = now;
                self.log(format!(
                    "[automation] Start template matched at ({},{},{},{}) active={} blue_ratio={:.3}",
                    button.rect.x, button.rect.y, button.rect.width, button.rect.height,
                    button.active, button.blue_ratio
                ));
            }
        } else {
            self.summary = format!(
                "START не найдена · OK {:.3} · прогон {}/{}",
                analysis.ok_button.score,
                self.completed_runs.saturating_add(1).min(self.runs),
                self.runs
            );
        }

        match phase {
            Some(AutomationPhase::WaitingForStart) => {
                if analysis.start_button.as_ref().is_some_and(|button| button.active) {
                    self.consecutive_matches = self.consecutive_matches.saturating_add(1);
                } else {
                    self.consecutive_matches = 0;
                }

                if self.consecutive_matches >= 2 {
                    if f12_pressed() {
                        self.stop_automation("Аварийная остановка: нажата F12.");
                        return;
                    }
                    let Some(button) = analysis.start_button else {
                        self.consecutive_matches = 0;
                        return;
                    };
                    let x = button.rect.x + button.rect.width / 2;
                    let y = button.rect.y + button.rect.height / 2;
                    match click_window_point(handle, x, y) {
                        Ok(()) => {
                            self.log(format!(
                                "[automation] START found and clicked at ({x}, {y}) using template"
                            ));
                            self.automation_phase = Some(AutomationPhase::WaitingForOk);
                            self.phase_started = Instant::now();
                            self.consecutive_matches = 0;
                            self.next_automation_poll =
                                Instant::now() + Duration::from_millis(550);
                            self.status = format!(
                                "Прогон {}: зачарование запущено; ожидание кнопки OK.",
                                self.completed_runs + 1
                            );
                        }
                        Err(error) => {
                            self.stop_automation(&format!("Клик по START не выполнен: {error}"));
                        }
                    }
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("[automation] waiting for stable blue Start-button match");
                }
            }
            Some(AutomationPhase::WaitingForOk) => {
                if analysis.ok_button.found {
                    self.consecutive_matches = self.consecutive_matches.saturating_add(1);
                } else {
                    self.consecutive_matches = 0;
                }

                if self.consecutive_matches >= 2 {
                    if f12_pressed() {
                        self.stop_automation("Аварийная остановка: нажата F12.");
                        return;
                    }
                    let Some(rect) = analysis.ok_button.rect else {
                        self.consecutive_matches = 0;
                        return;
                    };
                    let x = rect.x + rect.width / 2;
                    let y = rect.y + rect.height / 2;
                    match click_window_point(handle, x, y) {
                        Ok(()) => {
                            self.completed_runs = self.completed_runs.saturating_add(1);
                            if let Some(item) = &self.current_item {
                                self.attempted_items.insert((item.row, item.col));
                            }
                            self.log(format!(
                                "[automation] clicked OK at ({x}, {y}); completed {}/{}",
                                self.completed_runs, self.runs
                            ));
                            self.consecutive_matches = 0;
                            self.automation_phase = Some(AutomationPhase::ReconcileInventory);
                            self.phase_started = Instant::now();
                            self.next_automation_poll =
                                Instant::now() + Duration::from_millis(750);
                            self.status = "OK подтверждена. Повторное сканирование инвентаря и обновление логической таблицы.".into();
                        }
                        Err(error) => {
                            self.stop_automation(&format!("Клик по OK не выполнен: {error}"));
                        }
                    }
                } else if now.duration_since(self.last_wait_log) >= Duration::from_secs(10) {
                    self.last_wait_log = now;
                    self.log("[automation] waiting for stable OK template match");
                }
            }
            _ => {}
        }
    }

    fn logical_inventory_ui(&self, ui: &mut egui::Ui) {
        if self.logical_inventory.is_empty() {
            ui.label("Таблица пока не построена. Она создаётся при нажатии «Начать зачарование».");
            return;
        }

        ui.label(format!(
            "{} строк × {} столбцов · значок ✓ — уже обработанный предмет",
            self.inventory.rows, self.inventory.columns
        ));
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
                                ("A", egui::Color32::YELLOW, "Амулет")
                            }
                            Some(cell) if cell.kind == CellKind::Occupied
                                && self.attempted_items.contains(&(row, col)) =>
                            {
                                ("✓", egui::Color32::LIGHT_GRAY, "Уже обработан; повторно не нажимается")
                            }
                            Some(cell) if cell.kind == CellKind::Occupied => {
                                ("●", egui::Color32::LIGHT_GREEN, "Предмет доступен")
                            }
                            Some(cell) if cell.kind == CellKind::BeforeStart => {
                                ("—", egui::Color32::GRAY, "До стартовой позиции")
                            }
                            Some(cell) if cell.kind == CellKind::OutOfFrame => {
                                ("?", egui::Color32::GRAY, "Вне кадра")
                            }
                            Some(_) => ("·", egui::Color32::LIGHT_RED, "Пусто"),
                            None => ("?", egui::Color32::GRAY, "Ячейка не просканирована"),
                        };
                        ui.colored_label(color, egui::RichText::new(symbol).strong().monospace())
                            .on_hover_text(format!("Строка {}, столбец {}: {hint}", row + 1, col + 1));
                    }
                    ui.end_row();
                }
            });
    }

    fn inventory_ui(&mut self, ui: &mut egui::Ui) {
        let config = &mut self.inventory;
        if self.inventory_grid_region.is_configured() {
            ui.label(format!(
                "Область сетки: ({}, {}) · {}×{} px",
                self.inventory_grid_region.x,
                self.inventory_grid_region.y,
                self.inventory_grid_region.width,
                self.inventory_grid_region.height
            ));
        } else {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Область сетки не задана: выделите всю сетку на превью игрового окна.",
            );
        }
        egui::Grid::new("grid_cfg").num_columns(4).show(ui, |ui| {
            ui.label("X");
            ui.add(egui::DragValue::new(&mut config.origin_x).range(0..=3000));
            ui.label("Y");
            ui.add(egui::DragValue::new(&mut config.origin_y).range(0..=3000));
            ui.end_row();

            let previous_columns = config.columns;
            let previous_rows = config.rows;
            ui.label("Столбцы");
            ui.add(egui::DragValue::new(&mut config.columns).range(1..=12));
            ui.label("Строки");
            ui.add(egui::DragValue::new(&mut config.rows).range(1..=30));
            ui.end_row();
            if config.columns != previous_columns || config.rows != previous_rows {
                self.inventory_grid_region = Region::default();
                self.status = "Число строк/столбцов изменено. Повторно выделите сетку инвентаря.".into();
            }

            ui.label("Ячейка W/H");
            ui.add(egui::DragValue::new(&mut config.cell_width).range(8..=200));
            ui.add(egui::DragValue::new(&mut config.cell_height).range(8..=200));
            ui.end_row();

            ui.label("Шаг X/Y");
            ui.add(egui::DragValue::new(&mut config.step_x).range(8..=250));
            ui.add(egui::DragValue::new(&mut config.step_y).range(8..=250));
            ui.end_row();

            ui.label("Отступ");
            ui.add(egui::DragValue::new(&mut config.inset).range(0..=30));
            ui.label("Порог");
            ui.add(egui::Slider::new(&mut config.threshold, 0.0..=1.0).step_by(0.01));
            ui.end_row();
        });

        ui.horizontal(|ui| {
            ui.label("Метод");
            egui::ComboBox::from_id_salt("method")
                .selected_text(config.method.label())
                .show_ui(ui, |ui| {
                    for method in crate::vision::DetectionMethod::ALL {
                        ui.selectable_value(&mut config.method, method, method.label());
                    }
                });
        });
        ui.horizontal(|ui| {
            let mut start_row = config.start_row + 1;
            let mut start_col = config.start_col + 1;
            ui.label("Старт строка/столбец");
            ui.add(egui::DragValue::new(&mut start_row).range(1..=config.rows.max(1)));
            ui.add(egui::DragValue::new(&mut start_col).range(1..=config.columns.max(1)));
            config.start_row = start_row.saturating_sub(1).min(config.rows.saturating_sub(1));
            config.start_col = start_col.saturating_sub(1).min(config.columns.saturating_sub(1));
        });
        ui.horizontal(|ui| {
            let mut amulet_row = config.amulet_row + 1;
            let mut amulet_col = config.amulet_col + 1;
            ui.label("Амулет строка/столбец");
            ui.add(egui::DragValue::new(&mut amulet_row).range(1..=config.rows.max(1)));
            ui.add(egui::DragValue::new(&mut amulet_col).range(1..=config.columns.max(1)));
            config.amulet_row = amulet_row.saturating_sub(1).min(config.rows.saturating_sub(1));
            config.amulet_col = amulet_col.saturating_sub(1).min(config.columns.saturating_sub(1));
        });
        ui.label("Зелёный = предмет; оранжевый = амулет; серый = до старта; красный = пусто.");
        ui.collapsing("Пороги Canny", |ui| {
            ui.add(egui::Slider::new(&mut config.canny_low, 0.0..=200.0).text("low"));
            ui.add(egui::Slider::new(&mut config.canny_high, 10.0..=300.0).text("high"));
        });
    }

    fn enchant_ui(&mut self, ui: &mut egui::Ui) {
        let config = &mut self.enchant;
        ui.add(
            egui::Slider::new(&mut config.confidence_threshold, 0.50..=0.99)
                .step_by(0.01)
                .text("Порог совпадения OK"),
        );
        ui.horizontal(|ui| {
            ui.label("Масштаб");
            ui.add(egui::DragValue::new(&mut config.scale_min).speed(0.01).range(0.5..=2.0));
            ui.label("—");
            ui.add(egui::DragValue::new(&mut config.scale_max).speed(0.01).range(0.5..=2.0));
            ui.label("шаг");
            ui.add(egui::DragValue::new(&mut config.scale_step).speed(0.01).range(0.01..=0.25));
        });
        if config.scale_min > config.scale_max {
            std::mem::swap(&mut config.scale_min, &mut config.scale_max);
        }

        ui.separator();
        ui.label(egui::RichText::new("Кнопка «Начать»").strong());
        ui.label("Сначала ищется заголовок окна «Зачарование», затем внутри него — шаблон кнопки «Начать». Это отличает Start от соседних «Купить» и «Закрыть»; синий цвет подтверждает готовность.");
        ui.add(
            egui::Slider::new(&mut config.start_template_threshold, 0.45..=0.98)
                .step_by(0.01)
                .text("Порог совпадения шаблона START"),
        );
        ui.add(
            egui::Slider::new(&mut config.min_blue_ratio, 0.05..=0.90)
                .step_by(0.01)
                .text("Минимальная доля синего"),
        );
        let region = &mut config.start_button_region;
        egui::Grid::new("start_roi").num_columns(4).show(ui, |ui| {
            ui.label("X");
            ui.add(egui::DragValue::new(&mut region.x).range(0..=10000));
            ui.label("Y");
            ui.add(egui::DragValue::new(&mut region.y).range(0..=10000));
            ui.end_row();
            ui.label("Ширина");
            ui.add(egui::DragValue::new(&mut region.width).range(0..=3000));
            ui.label("Высота");
            ui.add(egui::DragValue::new(&mut region.height).range(0..=2000));
            ui.end_row();
        });
        if let Some(region) = self.last_enchant_window {
            ui.small(format!(
                "Последнее найденное окно зачарования: ({}, {}) · {}×{} px",
                region.x, region.y, region.width, region.height
            ));
        }
        if let Some(region) = self.last_start_button {
            ui.small(format!(
                "Последнее найденное положение START: ({}, {}) · {}×{} px",
                region.x, region.y, region.width, region.height
            ));
        }
        if !region.is_configured() {
            ui.label("Ручная область не задана — используется автоматический поиск по шаблону.");
        } else {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Ручная область: x={} y={} · {}×{} px",
                    region.x, region.y, region.width, region.height
                ));
                if ui.small_button("Сбросить и искать шаблоном").clicked() {
                    *region = Region::default();
                }
            });
        }
    }

    fn game_window_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label("Игровое окно");
            let selection_enabled = self.automation_phase.is_none();
            let previous_window = self.selected_window;
            ui.add_enabled_ui(selection_enabled, |ui| {
                let selected = self.selected_label();
                egui::ComboBox::from_id_salt("game_window")
                    .width((ui.available_width() - 100.0).max(180.0))
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        for window in &self.windows {
                            let label = format!(
                                "{} · PID {} · HWND 0x{:X} · {}×{}",
                                window.title, window.process_id, window.handle, window.width, window.height
                            );
                            ui.selectable_value(
                                &mut self.selected_window,
                                Some(window.handle),
                                label,
                            );
                        }
                    });
                if ui.button("↻").on_hover_text("Обновить список окон").clicked() {
                    self.refresh_windows(ctx);
                }
            });
            if !selection_enabled {
                ui.small("Выбор блокирован во время выполнения");
            }
            if self.selected_window != previous_window {
                self.inventory_grid_region = Region::default();
                self.logical_inventory.clear();
                self.attempted_items.clear();
                self.amulet_slot = None;
                self.current_item = None;
                self.enchant.start_button_region = Region::default();
                self.last_start_button = None;
                self.last_enchant_window = None;
                self.status = "Окно изменено. Повторно выделите сетку инвентаря; START будет найден по шаблону.".into();
                self.capture_selected_preview(ctx);
                self.log("[window] target changed; old ROI coordinates cleared");
            }
        });

        if self.selected_window_data().is_none() {
            ui.label("Выберите открытое окно игры из списка.");
            return;
        }

        let can_calibrate = self.automation_phase.is_none();
        ui.add_enabled_ui(can_calibrate, |ui| {
            ui.horizontal(|ui| {
                let start_changed = ui
                    .checkbox(&mut self.select_start_region, "Ручная область START")
                    .changed();
                if start_changed && self.select_start_region {
                    self.select_inventory_region = false;
                }
                let inventory_changed = ui
                    .checkbox(&mut self.select_inventory_region, "Область сетки инвентаря")
                    .changed();
                if inventory_changed && self.select_inventory_region {
                    self.select_start_region = false;
                }
                if ui.button("Обновить кадр").clicked() {
                    self.capture_selected_preview(ctx);
                }
            });
        });
        if can_calibrate {
            ui.small("Окно зачарования и START ищутся по шаблонам автоматически. Для инвентаря пока требуется один раз выделить всю сетку от первой до последней ячейки.");
        } else {
            ui.small("Калибровка заблокирована до завершения или остановки серии.");
        }

        if let Some(handle) = self.selected_window {
            if let Some(texture) = self.window_previews.get(&handle) {
                let raw_size = texture.size_vec2();
                let scale = (ui.available_width() / raw_size.x.max(1.0))
                    .min(520.0 / raw_size.y.max(1.0))
                    .min(1.0);
                let image_size = raw_size * scale;
                let response = ui.add(
                    egui::Image::new((texture.id(), image_size))
                        .sense(if can_calibrate && (self.select_start_region || self.select_inventory_region) {
                            egui::Sense::click_and_drag()
                        } else {
                            egui::Sense::hover()
                        }),
                );

                if can_calibrate
                    && (self.select_start_region || self.select_inventory_region)
                    && response.drag_started()
                {
                    self.region_drag_start = response.interact_pointer_pos();
                }
                if can_calibrate
                    && (self.select_start_region || self.select_inventory_region)
                    && response.dragged()
                    && self.region_drag_start.is_none()
                {
                    self.region_drag_start = response.interact_pointer_pos();
                }
                if can_calibrate
                    && (self.select_start_region || self.select_inventory_region)
                    && response.drag_stopped()
                {
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
                            let x = ((point.x - response.rect.left()) / response.rect.width())
                                * raw_size.x;
                            let y = ((point.y - response.rect.top()) / response.rect.height())
                                * raw_size.y;
                            (x.round() as i32, y.round() as i32)
                        };
                        let (x1, y1) = to_image(egui::pos2(start.x.min(end.x), start.y.min(end.y)));
                        let (x2, y2) = to_image(egui::pos2(start.x.max(end.x), start.y.max(end.y)));
                        if x2 - x1 >= 12 && y2 - y1 >= 8 {
                            if self.select_inventory_region {
                                let width = x2 - x1;
                                let height = y2 - y1;
                                let columns = f64::from(self.inventory.columns.max(1));
                                let rows = f64::from(self.inventory.rows.max(1));
                                // The drag rectangle encloses the outer bounds of all cells.
                                // Existing scanner geometry uses cells at 90% of the grid step.
                                let step_x = (f64::from(width) / (columns - 1.0 + 0.9))
                                    .floor() as i32;
                                let step_y = (f64::from(height) / (rows - 1.0 + 0.9))
                                    .floor() as i32;
                                if step_x < 8 || step_y < 8 {
                                    self.status = "Выбранная область слишком мала для заданного числа строк и столбцов.".into();
                                } else {
                                    self.inventory.origin_x = x1;
                                    self.inventory.origin_y = y1;
                                    self.inventory.step_x = step_x;
                                    self.inventory.step_y = step_y;
                                    self.inventory.cell_width =
                                        ((f64::from(step_x) * 0.9).floor() as i32)
                                            .clamp(8, step_x);
                                    self.inventory.cell_height =
                                        ((f64::from(step_y) * 0.9).floor() as i32)
                                            .clamp(8, step_y);
                                    self.inventory_grid_region = Region {
                                        x: x1,
                                        y: y1,
                                        width,
                                        height,
                                    };
                                    self.select_inventory_region = false;
                                    self.mode = Mode::Inventory;
                                    self.status = format!(
                                        "Сетка инвентаря откалибрована: {}×{}; шаг {}×{} px.",
                                        self.inventory.columns,
                                        self.inventory.rows,
                                        self.inventory.step_x,
                                        self.inventory.step_y
                                    );
                                    self.log(format!(
                                        "[ui] inventory grid ROI=({x1},{y1},{width},{height}) step={}x{} cell={}x{}",
                                        self.inventory.step_x,
                                        self.inventory.step_y,
                                        self.inventory.cell_width,
                                        self.inventory.cell_height
                                    ));
                                }
                            } else if self.select_start_region {
                                self.enchant.start_button_region = Region {
                                    x: x1,
                                    y: y1,
                                    width: x2 - x1,
                                    height: y2 - y1,
                                };
                                self.mode = Mode::Enchant;
                                self.select_start_region = false;
                                self.status = "Ручная область кнопки START сохранена; для возврата к поиску по обученному шаблону сбросьте ROI.".into();
                                self.log(format!(
                                    "[ui] manual START ROI=({x1},{y1},{},{})",
                                    x2 - x1,
                                    y2 - y1
                                ));
                            }
                        } else {
                            self.status = "Область слишком мала. Выделите прямоугольник целиком.".into();
                        }
                    }
                }
                let draw_region = |region: Region, color: egui::Color32| {
                    if !region.is_configured() || raw_size.x <= 0.0 || raw_size.y <= 0.0 {
                        return;
                    }
                    let min = egui::pos2(
                        response.rect.left()
                            + (region.x as f32 / raw_size.x) * response.rect.width(),
                        response.rect.top()
                            + (region.y as f32 / raw_size.y) * response.rect.height(),
                    );
                    let max = egui::pos2(
                        response.rect.left()
                            + ((region.x + region.width) as f32 / raw_size.x) * response.rect.width(),
                        response.rect.top()
                            + ((region.y + region.height) as f32 / raw_size.y) * response.rect.height(),
                    );
                    let stroke = egui::Stroke::new(2.0, color);
                    let painter = ui.painter();
                    painter.line_segment([min, egui::pos2(max.x, min.y)], stroke);
                    painter.line_segment([egui::pos2(max.x, min.y), max], stroke);
                    painter.line_segment([max, egui::pos2(min.x, max.y)], stroke);
                    painter.line_segment([egui::pos2(min.x, max.y), min], stroke);
                };
                draw_region(self.inventory_grid_region, egui::Color32::LIGHT_BLUE);
                draw_region(self.enchant.start_button_region, egui::Color32::GREEN);
                ui.small(format!(
                    "Размер кадра: {}×{} px · координаты ROI задаются в пикселях исходного окна",
                    raw_size.x as u32, raw_size.y as u32
                ));
            } else {
                ui.label(
                    self.window_preview_errors
                        .get(&handle)
                        .map(String::as_str)
                        .unwrap_or("Нажмите «Обновить кадр» для получения превью."),
                );
            }
        }
    }
}

impl eframe::App for EnchantmentHelperApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_automation(ctx);

        if Instant::now() >= self.next_window_refresh && self.selected_window.is_some() {
            self.capture_selected_preview(ctx);
            self.next_window_refresh = Instant::now() + Duration::from_secs(2);
        }

        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("JD Enchantment Helper");
                ui.separator();
                let state_text = match self.run_state {
                    RunState::Running => "АВТОЗАЧАРОВАНИЕ",
                    RunState::Paused => "ПАУЗА",
                    RunState::Stopped => "ОСТАНОВЛЕНО",
                    RunState::Idle => "ГОТОВО",
                };
                ui.label(state_text);
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
                        ui.label(egui::RichText::new("Прогоны").strong());
                        if ui
                            .add_enabled(
                                self.run_state != RunState::Running
                                    && self.run_state != RunState::Paused
                                    && self.runs > 1,
                                egui::Button::new("◀"),
                            )
                            .clicked()
                        {
                            self.runs = self.runs.saturating_sub(1).max(1);
                        }
                        ui.label(
                            egui::RichText::new(self.runs.to_string())
                                .size(22.0)
                                .strong()
                                .monospace(),
                        );
                        if ui
                            .add_enabled(
                                self.run_state != RunState::Running
                                    && self.run_state != RunState::Paused
                                    && self.runs < MAX_RUNS,
                                egui::Button::new("▶"),
                            )
                            .clicked()
                        {
                            self.runs = self.runs.saturating_add(1).min(MAX_RUNS);
                        }

                        ui.separator();
                        if ui
                            .add_enabled(
                                self.automation_phase.is_none() && self.selected_window.is_some(),
                                egui::Button::new("Сканировать инвентарь"),
                            )
                            .clicked()
                        {
                            self.scan_inventory_manually(ctx);
                        }
                        if ui
                            .add_enabled(
                                self.run_state != RunState::Running
                                    && self.run_state != RunState::Paused,
                                egui::Button::new("Начать зачарование"),
                            )
                            .clicked()
                        {
                            self.start_automation(ctx);
                        }
                        if ui
                            .add_enabled(
                                self.run_state == RunState::Running,
                                egui::Button::new("Пауза"),
                            )
                            .clicked()
                        {
                            self.run_state = RunState::Paused;
                            self.paused_at = Some(Instant::now());
                            self.status = "Пауза. F12 по-прежнему останавливает автоматизацию.".into();
                        }
                        if ui
                            .add_enabled(
                                self.run_state == RunState::Paused,
                                egui::Button::new("Продолжить"),
                            )
                            .clicked()
                        {
                            if let Some(paused_at) = self.paused_at.take() {
                                self.phase_started += paused_at.elapsed();
                            }
                            self.run_state = RunState::Running;
                            self.next_automation_poll = Instant::now();
                        }
                        if ui
                            .add_enabled(
                                self.automation_phase.is_some()
                                    || self.run_state == RunState::Paused,
                                egui::Button::new("Стоп"),
                            )
                            .clicked()
                        {
                            self.stop_automation("Автоматизация остановлена пользователем.");
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label(format!(
                            "Выполнено: {} / {}",
                            self.completed_runs, self.runs
                        ));
                        ui.separator();
                        ui.label(self.run_state.label());
                    });
                });

                ui.group(|ui| {
                    ui.label(egui::RichText::new(&self.status).strong());
                    ui.label(&self.summary);
                });

                ui.collapsing("Логическая таблица инвентаря", |ui| {
                    self.logical_inventory_ui(ui);
                });

                ui.collapsing("Распознавание и калибровка", |ui| {
                    ui.add_enabled_ui(self.automation_phase.is_none(), |ui| {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut self.mode, Mode::Enchant, "Зачарование");
                            ui.selectable_value(&mut self.mode, Mode::Inventory, "Инвентарь");
                        });
                        match self.mode {
                            Mode::Enchant => self.enchant_ui(ui),
                            Mode::Inventory => self.inventory_ui(ui),
                        }
                    });
                    if self.automation_phase.is_some() {
                        ui.small("Пороговые значения и геометрия зафиксированы на время серии.");
                    }
                });

                ui.collapsing("Диагностика по файлу", |ui| {
                    ui.add_enabled_ui(self.automation_phase.is_none(), |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Путь");
                            ui.text_edit_singleline(&mut self.image_path);
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Анализировать файл").clicked() {
                                self.analyze_file(ctx);
                            }
                            ui.checkbox(&mut self.save_debug, "Сохранять debug PNG");
                            ui.label("Папка");
                            ui.text_edit_singleline(&mut self.output_dir);
                        });
                    });
                    if self.automation_phase.is_some() {
                        ui.small("Диагностика временно недоступна во время автоматизации.");
                    }
                });

                ui.columns(2, |columns| {
                    columns[0].heading("Текущий кадр / разметка");
                    columns[1].heading("Лог");
                    if let Some(texture) = &self.preview {
                        let scale = (columns[0].available_width() / self.preview_size.x.max(1.0))
                            .min(1.0);
                        columns[0].image((texture.id(), self.preview_size * scale));
                    } else {
                        columns[0].label("Превью игрового окна появится после выбора окна.");
                    }
                    egui::ScrollArea::vertical()
                        .max_height(280.0)
                        .stick_to_bottom(true)
                        .show(&mut columns[1], |ui| {
                            for line in self.logs.iter().rev().take(100).rev() {
                                ui.monospace(line);
                            }
                        });
                });
            });
        });

        // Keep the global F12 poll responsive even while the game window has focus.
        if self.automation_phase.is_some() {
            ctx.request_repaint_after(Duration::from_millis(35));
        } else if self.selected_window.is_some() {
            ctx.request_repaint_after(Duration::from_millis(500));
        }
    }
}
