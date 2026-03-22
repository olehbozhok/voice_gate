//! Voice Gate — entry point.

mod app;
mod audio;
mod config;
mod error;
mod inference;
mod pipeline;
mod speaker;
mod ui;
mod vad;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    log::info!("Voice Gate v{}", env!("CARGO_PKG_VERSION"));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Voice Gate")
            .with_inner_size([520.0, 560.0])
            .with_min_inner_size([400.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Voice Gate",
        options,
        Box::new(|cc| Ok(Box::new(app::VoiceGateApp::new(cc)))),
    )
}
