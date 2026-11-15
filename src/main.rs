mod app;
mod model;
mod platform;
mod vision;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 820.0])
            .with_min_inner_size([820.0, 640.0]),
        ..Default::default()
    };

    eframe::run_native(
        "JD Enchantment Helper",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::EnchantmentHelperApp::default()))),
    )
}
