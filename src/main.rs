mod app;
mod model;
mod platform;
mod settings;
mod vision;

fn main() -> eframe::Result<()> {
    let window_size = [980.0, 780.0];
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Помощник зачарования Jade Dynasty")
            .with_inner_size(window_size)
            .with_min_inner_size(window_size)
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "Помощник зачарования Jade Dynasty",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::EnchantmentHelperApp::default()))),
    )
}
