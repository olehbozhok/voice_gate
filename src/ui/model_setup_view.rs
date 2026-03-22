//! Model Setup screen — shown when required ONNX models are missing.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::models::{DownloadProgress, ModelStatus};

/// Action returned by the model setup view.
pub enum ModelSetupAction {
    None,
    /// User provided a custom models directory.
    SetModelsDir(PathBuf),
    /// User clicked "Download".
    Download,
}

/// Render the model setup screen.
pub fn show(
    ui: &mut egui::Ui,
    status: &ModelStatus,
    models_dir: &str,
    _download_progress: &Option<Arc<Mutex<DownloadProgress>>>,
) -> ModelSetupAction {
    let mut action = ModelSetupAction::None;

    ui.heading("Model Setup");
    ui.add_space(8.0);

    match status {
        ModelStatus::Missing(missing) => {
            ui.label("The following models are required but not found:");
            ui.add_space(4.0);

            for m in missing {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("  \u{2717}")
                            .color(egui::Color32::from_rgb(220, 60, 60)),
                    );
                    ui.label(&m.description);
                    ui.label(
                        egui::RichText::new(format!("({})", m.filename))
                            .weak(),
                    );
                });
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Option 1: Download
            ui.label(egui::RichText::new("Option 1: Download automatically").strong());
            ui.add_space(4.0);
            ui.label(format!("Models will be saved to: {}", models_dir));
            ui.add_space(4.0);
            if ui.button("Download Models").clicked() {
                action = ModelSetupAction::Download;
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            // Option 2: Browse
            ui.label(egui::RichText::new("Option 2: Specify model directory").strong());
            ui.add_space(4.0);
            ui.label("If you already have the models, point to their directory:");
            ui.add_space(4.0);
            if ui.button("Browse...").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    action = ModelSetupAction::SetModelsDir(path);
                }
            }
        }

        ModelStatus::Downloading {
            current_model,
            progress,
        } => {
            ui.label(format!("Downloading: {}", current_model));
            ui.add_space(8.0);

            let bar = egui::ProgressBar::new(*progress)
                .text(format!("{:.0}%", progress * 100.0))
                .animate(true);
            ui.add(bar);

            ui.add_space(4.0);
            ui.label(
                egui::RichText::new("Please wait, this may take a minute...")
                    .weak(),
            );
        }

        ModelStatus::DownloadComplete => {
            ui.label(
                egui::RichText::new("\u{2713} All models downloaded successfully!")
                    .color(egui::Color32::from_rgb(60, 180, 60)),
            );
            ui.add_space(4.0);
            ui.label("Restarting...");
        }

        ModelStatus::Error(err) => {
            ui.label(
                egui::RichText::new(format!("Download failed: {}", err))
                    .color(egui::Color32::from_rgb(220, 60, 60)),
            );
            ui.add_space(8.0);
            if ui.button("Retry Download").clicked() {
                action = ModelSetupAction::Download;
            }
        }

        ModelStatus::Ready => {
            // Should not reach here — app transitions away from setup screen.
        }
    }

    action
}
