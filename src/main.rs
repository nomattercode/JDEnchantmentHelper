mod app;
mod capture_worker;
mod model;
mod platform;
mod settings;
mod vision;

fn main() -> eframe::Result<()> {
    let window_size = [1240.0, 820.0];
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("Помощник зачарования Jade Dynasty")
            .with_inner_size(window_size)
            .with_min_inner_size([1180.0, 680.0])
            .with_resizable(true),
        ..Default::default()
    };

    eframe::run_native(
        "Помощник зачарования Jade Dynasty",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::EnchantmentHelperApp::default()))),
    )
}
