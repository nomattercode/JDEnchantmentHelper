mod app;
mod model;
mod platform;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([560.0, 390.0])
            .with_min_inner_size([480.0, 340.0]),
        ..Default::default()
    };

    eframe::run_native(
        "JD Enchantment Helper",
        options,
        Box::new(|_creation_context| Ok(Box::new(app::EnchantmentHelperApp::default()))),
    )
}
