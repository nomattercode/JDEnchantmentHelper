use eframe::egui;
use opencv::prelude::*;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::model::{GameWindow, RunState, MAX_RUNS};
use crate::platform::{
    capture_window, click_window_point, f12_pressed, list_visible_windows,
};
use crate::vision::{
    analyze_enchant_window, load_image, mat_to_rgb, save_png, scan_inventory,
    EnchantDetectorConfig, InventoryConfig, Region,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Inventory,
    Enchant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutomationPhase {
    WaitingForStart,
    WaitingForOk,
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
            status: "Выберите игровое окно и выделите область кнопки «Начать».".into(),
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
                    self.summary = format!(
                        "OK: {} · score {:.3} · {} · scale {:.2} · Start: {}",
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

    fn start_automation(&mut self) {
        if self.run_state == RunState::Running || self.run_state == RunState::Paused {
            return;
        }
        if self.selected_window.is_none() {
            self.status = "Сначала выберите окно игры.".into();
            return;
        }
        if !self.enchant.start_button_region.is_configured() {
            self.status = "Сначала выделите мышью прямоугольник кнопки «Начать» на превью окна.".into();
            self.select_start_region = true;
            self.mode = Mode::Enchant;
            return;
        }

        self.completed_runs = 0;
        self.consecutive_matches = 0;
        self.automation_phase = Some(AutomationPhase::WaitingForStart);
        self.run_state = RunState::Running;
        self.next_automation_poll = Instant::now();
        self.last_wait_log = Instant::now();
        self.status = format!(
            "Ожидание синей кнопки «Начать». F12 — аварийный стоп. План: {} прогон(ов).",
            self.runs
        );
        self.log(format!(
            "[automation] armed; target runs={} start ROI=({}, {}, {}, {})",
            self.runs,
            self.enchant.start_button_region.x,
            self.enchant.start_button_region.y,
            self.enchant.start_button_region.width,
            self.enchant.start_button_region.height
        ));
    }

    fn stop_automation(&mut self, reason: &str) {
        let was_active = self.automation_phase.is_some()
            || matches!(self.run_state, RunState::Running | RunState::Paused);
        self.automation_phase = None;
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

        let analysis = match analyze_enchant_window(&frame, &self.enchant) {
            Ok(analysis) => analysis,
            Err(error) => {
                self.stop_automation(&format!("Ошибка анализа OpenCV: {error}"));
                return;
            }
        };

        self.set_preview(ctx, &analysis.overlay);
        if let Some(button) = &analysis.start_button {
            self.summary = format!(
                "Start: {} · синяя область {:.0}% · OK score {:.3} · прогон {}/{}",
                if button.active { "активна" } else { "неактивна" },
                button.blue_ratio * 100.0,
                analysis.ok_button.score,
                self.completed_runs.saturating_add(1).min(self.runs),
                self.runs
            );
        } else {
            self.summary = format!(
                "Start не найдена · OK score {:.3} · прогон {}/{}",
                analysis.ok_button.score,
                self.completed_runs.saturating_add(1).min(self.runs),
                self.runs
            );
        }

        match self.automation_phase {
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
                                "[automation] clicked START at ({x}, {y}); waiting for OK"
                            ));
                            self.automation_phase = Some(AutomationPhase::WaitingForOk);
                            self.consecutive_matches = 0;
                            self.next_automation_poll = Instant::now() + Duration::from_millis(500);
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
                    self.log("[automation] waiting for active blue START button (ROI only)");
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
                            self.log(format!(
                                "[automation] clicked OK at ({x}, {y}); completed {}/{}",
                                self.completed_runs, self.runs
                            ));
                            self.consecutive_matches = 0;
                            if self.completed_runs >= self.runs {
                                self.automation_phase = None;
                                self.run_state = RunState::Stopped;
                                self.status = format!(
                                    "Серия завершена: выполнено {} из {} прогонов.",
                                    self.completed_runs, self.runs
                                );
                            } else {
                                self.automation_phase = Some(AutomationPhase::WaitingForStart);
                                self.next_automation_poll =
                                    Instant::now() + Duration::from_millis(650);
                                self.status = format!(
                                    "OK подтверждена. Ожидание кнопки «Начать» для прогона {} из {}.",
                                    self.completed_runs + 1,
                                    self.runs
                                );
                            }
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
            None => {}
        }
    }

    fn inventory_ui(&mut self, ui: &mut egui::Ui) {
        let config = &mut self.inventory;
        egui::Grid::new("grid_cfg").num_columns(4).show(ui, |ui| {
            ui.label("X");
            ui.add(egui::DragValue::new(&mut config.origin_x).range(0..=3000));
            ui.label("Y");
            ui.add(egui::DragValue::new(&mut config.origin_y).range(0..=3000));
            ui.end_row();

            ui.label("Столбцы");
            ui.add(egui::DragValue::new(&mut config.columns).range(1..=12));
            ui.label("Строки");
            ui.add(egui::DragValue::new(&mut config.rows).range(1..=30));
            ui.end_row();

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
        ui.label("Выделите её область мышью на превью окна игры. Распознавание проверяет синие пиксели только внутри выделенного прямоугольника.");
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
        if !region.is_configured() {
            ui.colored_label(
                egui::Color32::YELLOW,
                "Область START не задана — автоматические клики пока заблокированы.",
            );
        } else {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "ROI: x={} y={} · {}×{} px",
                    region.x, region.y, region.width, region.height
                ));
                if ui.small_button("Сбросить").clicked() {
                    *region = Region::default();
                }
            });
        }
    }

    fn game_window_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            ui.label("Игровое окно");
            let selection_enabled = self.automation_phase.is_none();
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
        });

        if self.selected_window_data().is_none() {
            ui.label("Выберите открытое окно игры из списка.");
            return;
        }

        ui.horizontal(|ui| {
            ui.label("Перетащите мышью по кнопке «Начать» на превью ниже:");
            ui.checkbox(&mut self.select_start_region, "Выделить область");
            if ui.button("Обновить кадр").clicked() {
                self.capture_selected_preview(ctx);
            }
        });

        if let Some(handle) = self.selected_window {
            if let Some(texture) = self.window_previews.get(&handle) {
                let raw_size = texture.size_vec2();
                let scale = (ui.available_width() / raw_size.x.max(1.0))
                    .min(520.0 / raw_size.y.max(1.0))
                    .min(1.0);
                let image_size = raw_size * scale;
                let response = ui.add(
                    egui::Image::new((texture.id(), image_size))
                        .sense(if self.select_start_region {
                            egui::Sense::click_and_drag()
                        } else {
                            egui::Sense::hover()
                        }),
                );

                if self.select_start_region && response.drag_started() {
                    self.region_drag_start = response.interact_pointer_pos();
                }
                if self.select_start_region && response.dragged() {
                    if self.region_drag_start.is_none() {
                        self.region_drag_start = response.interact_pointer_pos();
                    }
                }
                if self.select_start_region && response.drag_stopped() {
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
                            self.enchant.start_button_region = Region {
                                x: x1,
                                y: y1,
                                width: x2 - x1,
                                height: y2 - y1,
                            };
                            self.mode = Mode::Enchant;
                            self.select_start_region = false;
                            self.status = "Область кнопки START сохранена. Проверьте рамку анализом тестового кадра перед первым запуском.".into();
                            self.log(format!(
                                "[ui] START ROI set to ({x1}, {y1}, {}, {})",
                                x2 - x1,
                                y2 - y1
                            ));
                        } else {
                            self.status = "Область слишком мала. Выделите кнопку целиком.".into();
                        }
                    }
                }
                let roi = self.enchant.start_button_region;
                if roi.is_configured() && raw_size.x > 0.0 && raw_size.y > 0.0 {
                    let min = egui::pos2(
                        response.rect.left()
                            + (roi.x as f32 / raw_size.x) * response.rect.width(),
                        response.rect.top()
                            + (roi.y as f32 / raw_size.y) * response.rect.height(),
                    );
                    let max = egui::pos2(
                        response.rect.left()
                            + ((roi.x + roi.width) as f32 / raw_size.x) * response.rect.width(),
                        response.rect.top()
                            + ((roi.y + roi.height) as f32 / raw_size.y) * response.rect.height(),
                    );
                    let stroke = egui::Stroke::new(2.0, egui::Color32::GREEN);
                    let painter = ui.painter();
                    painter.line_segment([min, egui::pos2(max.x, min.y)], stroke);
                    painter.line_segment([egui::pos2(max.x, min.y), max], stroke);
                    painter.line_segment([max, egui::pos2(min.x, max.y)], stroke);
                    painter.line_segment([egui::pos2(min.x, max.y), min], stroke);
                }
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
                                self.run_state != RunState::Running
                                    && self.run_state != RunState::Paused,
                                egui::Button::new("Начать зачарование"),
                            )
                            .clicked()
                        {
                            self.start_automation();
                        }
                        if ui
                            .add_enabled(
                                self.run_state == RunState::Running,
                                egui::Button::new("Пауза"),
                            )
                            .clicked()
                        {
                            self.run_state = RunState::Paused;
                            self.status = "Пауза. F12 по-прежнему останавливает автоматизацию.".into();
                        }
                        if ui
                            .add_enabled(
                                self.run_state == RunState::Paused,
                                egui::Button::new("Продолжить"),
                            )
                            .clicked()
                        {
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

                ui.collapsing("Распознавание и калибровка", |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.mode, Mode::Enchant, "Зачарование");
                        ui.selectable_value(&mut self.mode, Mode::Inventory, "Инвентарь");
                    });
                    match self.mode {
                        Mode::Enchant => self.enchant_ui(ui),
                        Mode::Inventory => self.inventory_ui(ui),
                    }
                });

                ui.collapsing("Диагностика по файлу", |ui| {
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
