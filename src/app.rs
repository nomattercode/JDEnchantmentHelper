use eframe::egui;

use crate::model::{GameWindow, RunState, MAX_RUNS};
use crate::platform::list_visible_windows;

pub struct EnchantmentHelperApp {
    windows: Vec<GameWindow>,
    selected_window: Option<isize>,
    runs_per_pass: u8,
    run_state: RunState,
    status_message: String,
}

impl Default for EnchantmentHelperApp {
    fn default() -> Self {
        let windows = list_visible_windows();

        Self {
            windows,
            selected_window: None,
            runs_per_pass: 1,
            run_state: RunState::Idle,
            status_message: "Готово к настройке. Автоматизация игровых действий ещё не подключена."
                .to_owned(),
        }
    }
}

impl EnchantmentHelperApp {
    fn refresh_windows(&mut self) {
        self.windows = list_visible_windows();

        if let Some(selected) = self.selected_window {
            if !self.windows.iter().any(|window| window.handle == selected) {
                self.selected_window = None;
            }
        }

        self.status_message = if self.windows.is_empty() {
            "Видимые окна с заголовками не найдены.".to_owned()
        } else {
            format!("Найдено окон: {}.", self.windows.len())
        };
    }

    fn selected_window_label(&self) -> String {
        self.selected_window
            .and_then(|selected| self.windows.iter().find(|window| window.handle == selected))
            .map(|window| format!("{} (PID: {})", window.title, window.process_id))
            .unwrap_or_else(|| "Окно не выбрано".to_owned())
    }
}

impl eframe::App for EnchantmentHelperApp {
    fn update(&mut self, context: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(context, |ui| {
            ui.heading("JD Enchantment Helper");
            ui.label("Начальная версия интерфейса");
            ui.add_space(8.0);
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Окно игры:");
                if ui.button("Обновить список").clicked() {
                    self.refresh_windows();
                }
            });

            let selected_label = self.selected_window_label();
            egui::ComboBox::from_id_source("game_window_selector")
                .selected_text(selected_label)
                .width(390.0)
                .show_ui(ui, |ui| {
                    for window in &self.windows {
                        let label = format!("{} (PID: {})", window.title, window.process_id);
                        ui.selectable_value(
                            &mut self.selected_window,
                            Some(window.handle),
                            label,
                        );
                    }
                });

            #[cfg(not(windows))]
            ui.colored_label(
                egui::Color32::YELLOW,
                "Перечисление окон игры доступно только в Windows.",
            );

            ui.add_space(12.0);
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Количество прогонов:");
                egui::ComboBox::from_id_source("run_count_selector")
                    .selected_text(self.runs_per_pass.to_string())
                    .show_ui(ui, |ui| {
                        for count in 1..=MAX_RUNS {
                            ui.selectable_value(
                                &mut self.runs_per_pass,
                                count,
                                count.to_string(),
                            );
                        }
                    });
                ui.label("из 10");
            });

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                let can_start = self.selected_window.is_some()
                    && self.run_state != RunState::Running;

                if ui
                    .add_enabled(can_start, egui::Button::new("Старт"))
                    .clicked()
                {
                    self.run_state = RunState::Running;
                    self.status_message =
                        "Интерфейс переведён в состояние запуска. Игровые действия пока не выполняются."
                            .to_owned();
                }

                if ui
                    .add_enabled(
                        self.run_state == RunState::Running,
                        egui::Button::new("Пауза"),
                    )
                    .clicked()
                {
                    self.run_state = RunState::Paused;
                    self.status_message = "Работа приостановлена.".to_owned();
                }

                if ui
                    .add_enabled(
                        matches!(self.run_state, RunState::Running | RunState::Paused),
                        egui::Button::new("Стоп"),
                    )
                    .clicked()
                {
                    self.run_state = RunState::Stopped;
                    self.status_message = "Работа остановлена.".to_owned();
                }
            });

            ui.add_space(10.0);
            ui.group(|ui| {
                ui.label(format!("Состояние: {}", self.run_state.label()));
                ui.label(format!("Выбрано прогонов: {}", self.runs_per_pass));
            });

            ui.add_space(8.0);
            ui.label(&self.status_message);

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.separator();
                ui.small(
                    "Каркас интерфейса: поиск предметов, захват экрана и управление игрой будут добавлены отдельно.",
                );
            });
        });
    }
}
