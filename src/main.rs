//! Voice Gate — entry point.
//!
//! ```bash
//! cargo run --release                   # CPU (NdArray, pure Rust)
//! cargo run --release --features wgpu   # GPU (Vulkan/Metal/DX12)
//! cargo run --release --features cuda   # NVIDIA CUDA
//! ```

mod app;
mod audio;
mod backend;
mod config;
mod error;
mod model;
mod pipeline;
mod speaker;
mod ui;
mod vad;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    log::info!("Voice Gate v{}", env!("CARGO_PKG_VERSION"));
    log::info!("Backend: {}", backend::backend_name());

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
