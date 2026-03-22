//! Settings — threshold sliders, backend info.

use egui::{RichText, Ui};
use crate::config::Config;

/// Returns true if anything changed.
pub fn show(ui: &mut Ui, config: &mut Config) -> bool {
    let mut changed = false;
    ui.heading("Settings");
    ui.add_space(8.0);

    ui.group(|ui| {
        ui.label(RichText::new("Voice Activity Detection").strong());
        ui.horizontal(|ui| {
            ui.label("Speech threshold:");
            if ui.add(egui::Slider::new(&mut config.vad.threshold, 0.1..=0.95).step_by(0.05)).changed() { changed = true; }
        });
        ui.label(egui::RichText::new("Higher = fewer false positives, may miss quiet speech.").small().weak());
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Speaker Verification").strong());
        ui.horizontal(|ui| {
            ui.label("Similarity threshold:");
            if ui.add(egui::Slider::new(&mut config.speaker.similarity_threshold, 0.40..=0.95).step_by(0.05)).changed() { changed = true; }
        });
        ui.label(egui::RichText::new("Lower = permissive. Higher = strict (may reject unusual intonation).").small().weak());
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Gate Behaviour").strong());
        ui.horizontal(|ui| {
            ui.label("Hold time (ms):");
            let mut hold = config.gate.hold_time_ms as f32;
            if ui.add(egui::Slider::new(&mut hold, 50.0..=1000.0).step_by(50.0)).changed() {
                config.gate.hold_time_ms = hold as u32; changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Pre-buffer (ms):");
            let mut pre = config.gate.pre_buffer_ms as f32;
            if ui.add(egui::Slider::new(&mut pre, 0.0..=500.0).step_by(25.0)).changed() {
                config.gate.pre_buffer_ms = pre as u32; changed = true;
            }
        });
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Runtime").strong());
        ui.label(format!("Inference backend: {}", crate::backend::backend_name()));
        #[cfg(has_silero_vad)]
        ui.label("Silero VAD: compiled (Burn native)");
        #[cfg(not(has_silero_vad))]
        ui.label("Silero VAD: NOT compiled (energy fallback)");
        #[cfg(has_ecapa_tdnn)]
        ui.label("ECAPA-TDNN: compiled (Burn native)");
        #[cfg(not(has_ecapa_tdnn))]
        ui.label("ECAPA-TDNN: NOT compiled (speaker verification disabled)");
    });

    changed
}
