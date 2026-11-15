use eframe::egui;
use opencv::prelude::*;

use crate::model::{GameWindow, RunState, MAX_RUNS};
use crate::platform::list_visible_windows;
use crate::vision::{
    analyze_enchant_window, load_image, mat_to_rgb, save_png, scan_inventory,
    EnchantDetectorConfig, InventoryConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode { Inventory, Enchant }

pub struct EnchantmentHelperApp {
    windows: Vec<GameWindow>,
    selected_window: Option<isize>,
    runs: u8,
    run_state: RunState,
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
}

impl Default for EnchantmentHelperApp {
    fn default() -> Self {
        Self {
            windows: list_visible_windows(), selected_window: None, runs: 1,
            run_state: RunState::Idle,
            status: "Диагностика по файлам. Управление игрой отключено.".into(),
            image_path: String::new(), output_dir: "diagnostics".into(), mode: Mode::Inventory,
            inventory: InventoryConfig::default(), enchant: EnchantDetectorConfig::default(),
            save_debug: true, preview: None, preview_size: egui::Vec2::ZERO,
            summary: "Изображение не загружено.".into(),
            logs: vec!["[app] OpenCV diagnostic mode ready".into()],
        }
    }
}

impl EnchantmentHelperApp {
    fn log(&mut self, line: impl Into<String>) {
        self.logs.push(line.into());
        if self.logs.len() > 300 { self.logs.drain(0..self.logs.len() - 300); }
    }

    fn selected_label(&self) -> String {
        self.selected_window.and_then(|h| self.windows.iter().find(|w| w.handle == h))
            .map(|w| format!("{} (PID {})", w.title, w.process_id))
            .unwrap_or_else(|| "Окно не выбрано".into())
    }

    fn set_preview(&mut self, ctx: &egui::Context, mat: &opencv::core::Mat) {
        match mat_to_rgb(mat) {
            Ok((w, h, pixels)) => {
                let image = egui::ColorImage::from_rgb([w, h], &pixels);
                self.preview_size = egui::vec2(w as f32, h as f32);
                if let Some(texture) = &mut self.preview {
                    texture.set(image, egui::TextureOptions::LINEAR);
                } else {
                    self.preview = Some(ctx.load_texture("vision-preview", image, egui::TextureOptions::LINEAR));
                }
            }
            Err(e) => self.log(format!("[preview] {e}")),
        }
    }

    fn save_debug(&mut self, name: &str, mat: &opencv::core::Mat) {
        if !self.save_debug { return; }
        let path = std::path::Path::new(&self.output_dir).join(name);
        match save_png(&path, mat) {
            Ok(()) => self.log(format!("[debug] saved {}", path.display())),
            Err(e) => self.log(format!("[debug] {e}")),
        }
    }

    fn analyze(&mut self, ctx: &egui::Context) {
        if self.image_path.trim().is_empty() {
            self.status = "Укажи путь к тестовому изображению.".into();
            return;
        }
        self.log(format!("[app] loading {}", self.image_path));
        let frame = match load_image(self.image_path.trim()) {
            Ok(m) => m,
            Err(e) => { self.status = e.clone(); self.log(format!("[error] {e}")); return; }
        };
        self.log(format!("[app] frame={}x{}", frame.cols(), frame.rows()));
        self.save_debug("input_frame.png", &frame);
        match self.mode {
            Mode::Inventory => match scan_inventory(&frame, &self.inventory) {
                Ok(r) => {
                    self.summary = format!("Ячеек: {} | кандидатов: {} | метод: {}", r.cells.len(), r.candidates.len(), self.inventory.method.label());
                    for line in &r.log { self.log(line.clone()); }
                    self.save_debug("inventory_overlay.png", &r.overlay);
                    self.save_debug("inventory_score_map.png", &r.score_map);
                    self.set_preview(ctx, &r.overlay);
                    self.status = "Сканирование завершено. Игровые действия не выполнялись.".into();
                }
                Err(e) => { self.status = e.to_string(); self.log(format!("[error][inventory] {e}")); }
            },
            Mode::Enchant => match analyze_enchant_window(&frame, &self.enchant) {
                Ok(r) => {
                    self.summary = format!("OK: {} | score {:.3} | {} | scale {:.2}", if r.ok_button.found { "найдена" } else { "не найдена" }, r.ok_button.score, r.ok_button.template_name, r.ok_button.scale);
                    for line in &r.log { self.log(line.clone()); }
                    self.save_debug("enchant_overlay.png", &r.overlay);
                    self.set_preview(ctx, &r.overlay);
                    self.status = "Анализ завершён. Игровые действия не выполнялись.".into();
                }
                Err(e) => { self.status = e.to_string(); self.log(format!("[error][enchant] {e}")); }
            },
        }
    }

    fn inventory_ui(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.inventory;
        egui::Grid::new("grid_cfg").num_columns(4).show(ui, |ui| {
            ui.label("X"); ui.add(egui::DragValue::new(&mut c.origin_x).range(0..=3000));
            ui.label("Y"); ui.add(egui::DragValue::new(&mut c.origin_y).range(0..=3000)); ui.end_row();
            ui.label("Столбцы"); ui.add(egui::DragValue::new(&mut c.columns).range(1..=12));
            ui.label("Строки"); ui.add(egui::DragValue::new(&mut c.rows).range(1..=30)); ui.end_row();
            ui.label("Ячейка W/H"); ui.add(egui::DragValue::new(&mut c.cell_width).range(8..=200));
            ui.add(egui::DragValue::new(&mut c.cell_height).range(8..=200)); ui.end_row();
            ui.label("Шаг X/Y"); ui.add(egui::DragValue::new(&mut c.step_x).range(8..=250));
            ui.add(egui::DragValue::new(&mut c.step_y).range(8..=250)); ui.end_row();
            ui.label("Отступ"); ui.add(egui::DragValue::new(&mut c.inset).range(0..=30));
            ui.label("Порог"); ui.add(egui::Slider::new(&mut c.threshold, 0.0..=1.0).step_by(0.01)); ui.end_row();
        });
        ui.horizontal(|ui| {
            ui.label("Метод");
            egui::ComboBox::from_id_salt("method").selected_text(c.method.label()).show_ui(ui, |ui| {
                for m in crate::vision::DetectionMethod::ALL { ui.selectable_value(&mut c.method, m, m.label()); }
            });
        });
        ui.horizontal(|ui| {
            let mut sr = c.start_row + 1; let mut sc = c.start_col + 1;
            ui.label("Старт строка/столбец"); ui.add(egui::DragValue::new(&mut sr).range(1..=c.rows.max(1)));
            ui.add(egui::DragValue::new(&mut sc).range(1..=c.columns.max(1)));
            c.start_row = sr.saturating_sub(1).min(c.rows.saturating_sub(1));
            c.start_col = sc.saturating_sub(1).min(c.columns.saturating_sub(1));
        });
        ui.horizontal(|ui| {
            let mut ar = c.amulet_row + 1; let mut ac = c.amulet_col + 1;
            ui.label("Амулет строка/столбец"); ui.add(egui::DragValue::new(&mut ar).range(1..=c.rows.max(1)));
            ui.add(egui::DragValue::new(&mut ac).range(1..=c.columns.max(1)));
            c.amulet_row = ar.saturating_sub(1).min(c.rows.saturating_sub(1));
            c.amulet_col = ac.saturating_sub(1).min(c.columns.saturating_sub(1));
        });
        ui.label("Зелёный = предмет; оранжевый = амулет; серый = до старта; красный = пусто.");
        ui.collapsing("Пороги Canny", |ui| {
            ui.add(egui::Slider::new(&mut c.canny_low, 0.0..=200.0).text("low"));
            ui.add(egui::Slider::new(&mut c.canny_high, 10.0..=300.0).text("high"));
        });
    }

    fn enchant_ui(&mut self, ui: &mut egui::Ui) {
        let c = &mut self.enchant;
        ui.add(egui::Slider::new(&mut c.confidence_threshold, 0.50..=0.99).step_by(0.01).text("Порог совпадения OK"));
        ui.horizontal(|ui| {
            ui.label("Масштаб"); ui.add(egui::DragValue::new(&mut c.scale_min).speed(0.01).range(0.5..=2.0));
            ui.label("—"); ui.add(egui::DragValue::new(&mut c.scale_max).speed(0.01).range(0.5..=2.0));
            ui.label("шаг"); ui.add(egui::DragValue::new(&mut c.scale_step).speed(0.01).range(0.01..=0.25));
        });
        if c.scale_min > c.scale_max { std::mem::swap(&mut c.scale_min, &mut c.scale_max); }
        ui.checkbox(&mut c.check_start_button, "Оценивать активность кнопки «Вставить» (ROI нужно откалибровать)");
        if c.check_start_button {
            egui::Grid::new("start_roi").num_columns(4).show(ui, |ui| {
                ui.label("X"); ui.add(egui::DragValue::new(&mut c.start_button_region.x).range(0..=3000));
                ui.label("Y"); ui.add(egui::DragValue::new(&mut c.start_button_region.y).range(0..=3000)); ui.end_row();
                ui.label("W"); ui.add(egui::DragValue::new(&mut c.start_button_region.width).range(8..=500));
                ui.label("H"); ui.add(egui::DragValue::new(&mut c.start_button_region.height).range(8..=300)); ui.end_row();
            });
            ui.add(egui::Slider::new(&mut c.active_saturation_threshold, 0.0..=1.0).text("Порог насыщенности"));
        }
    }
}

impl eframe::App for EnchantmentHelperApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("JD Enchantment Helper"); ui.separator();
                ui.colored_label(egui::Color32::YELLOW, "DIAGNOSTIC MODE — GAME INPUT DISABLED");
            });
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.collapsing("Игровое окно и прогоны", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Окно");
                        if ui.button("Обновить").clicked() {
                            self.windows = list_visible_windows();
                            self.log(format!("[window] visible windows={}", self.windows.len()));
                        }
                        egui::ComboBox::from_id_salt("window").selected_text(self.selected_label()).show_ui(ui, |ui| {
                            for w in &self.windows { ui.selectable_value(&mut self.selected_window, Some(w.handle), format!("{} (PID {})", w.title, w.process_id)); }
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Прогонов");
                        egui::ComboBox::from_id_salt("runs").selected_text(self.runs.to_string()).show_ui(ui, |ui| {
                            for n in 1..=MAX_RUNS { ui.selectable_value(&mut self.runs, n, n.to_string()); }
                        });
                        if ui.button("Старт").clicked() { self.run_state = RunState::Running; }
                        if ui.add_enabled(self.run_state == RunState::Running, egui::Button::new("Пауза")).clicked() { self.run_state = RunState::Paused; }
                        if ui.add_enabled(matches!(self.run_state, RunState::Running | RunState::Paused), egui::Button::new("Стоп")).clicked() { self.run_state = RunState::Stopped; }
                        ui.label(self.run_state.label());
                    });
                });
                ui.separator();
                ui.heading("Тестирование OpenCV на изображении");
                ui.horizontal(|ui| { ui.label("Путь"); ui.text_edit_singleline(&mut self.image_path); });
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.mode, Mode::Inventory, "Инвентарь");
                    ui.selectable_value(&mut self.mode, Mode::Enchant, "Окно зачарования / OK");
                    if ui.button("Анализировать").clicked() { self.analyze(ctx); }
                    ui.checkbox(&mut self.save_debug, "Сохранять debug PNG");
                });
                ui.collapsing("Параметры распознавания", |ui| {
                    match self.mode { Mode::Inventory => self.inventory_ui(ui), Mode::Enchant => self.enchant_ui(ui) }
                });
                ui.horizontal(|ui| { ui.label("Папка диагностики"); ui.text_edit_singleline(&mut self.output_dir); });
                ui.group(|ui| { ui.label(egui::RichText::new(&self.summary).strong()); ui.label(&self.status); });
                ui.columns(2, |cols| {
                    cols[0].heading("Визуальная разметка"); cols[1].heading("Лог OpenCV");
                    if let Some(texture) = &self.preview {
                        let scale = (cols[0].available_width() / self.preview_size.x.max(1.0)).min(1.0);
                        cols[0].image((texture.id(), self.preview_size * scale));
                    } else { cols[0].label("Выбери изображение и нажми «Анализировать»."); }
                    egui::ScrollArea::vertical().max_height(440.0).stick_to_bottom(true).show(&mut cols[1], |ui| {
                        for line in self.logs.iter().rev().take(100).rev() { ui.monospace(line); }
                    });
                });
                ui.small("Кадр загружается из файла. Захват игрового окна и синтетические клики ещё не подключены.");
            });
        });
    }
}
