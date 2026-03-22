//! Settings — threshold sliders, backend info.

use egui::{RichText, Ui};
use crate::config::{Config, GateMode};

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
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Verification mode:");
            if ui.selectable_label(config.gate.mode == GateMode::Optimistic, "Optimistic").clicked() {
                config.gate.mode = GateMode::Optimistic; changed = true;
            }
            if ui.selectable_label(config.gate.mode == GateMode::Strict, "Strict").clicked() {
                config.gate.mode = GateMode::Strict; changed = true;
            }
        });
        let mode_hint = match config.gate.mode {
            GateMode::Optimistic => "Opens instantly, closes if not owner (~1.5s check).",
            GateMode::Strict => "Stays closed until speaker verified (~1.5s delay).",
        };
        ui.label(egui::RichText::new(mode_hint).small().weak());
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Audio Devices").strong());
        ui.label(egui::RichText::new("Changes take effect on next Start.").small().weak());
        ui.add_space(4.0);

        // Input device
        let input_devices = crate::audio::capture::list_input_devices();
        let current_input = config.audio.input_device.clone().unwrap_or_else(|| "(System Default)".to_string());
        ui.horizontal(|ui| {
            ui.label("Input:");
            egui::ComboBox::from_id_salt("input_device")
                .selected_text(&current_input)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(config.audio.input_device.is_none(), "(System Default)").clicked() {
                        config.audio.input_device = None;
                        changed = true;
                    }
                    for name in &input_devices {
                        let selected = config.audio.input_device.as_deref() == Some(name.as_str());
                        if ui.selectable_label(selected, name).clicked() {
                            config.audio.input_device = Some(name.clone());
                            changed = true;
                        }
                    }
                });
        });

        // Output device
        let output_devices = crate::audio::output::list_output_devices();
        let current_output = config.audio.output_device.clone().unwrap_or_else(|| "(System Default)".to_string());
        ui.horizontal(|ui| {
            ui.label("Output:");
            egui::ComboBox::from_id_salt("output_device")
                .selected_text(&current_output)
                .show_ui(ui, |ui| {
                    if ui.selectable_label(config.audio.output_device.is_none(), "(System Default)").clicked() {
                        config.audio.output_device = None;
                        changed = true;
                    }
                    for name in &output_devices {
                        let selected = config.audio.output_device.as_deref() == Some(name.as_str());
                        if ui.selectable_label(selected, name).clicked() {
                            config.audio.output_device = Some(name.clone());
                            changed = true;
                        }
                    }
                });
        });
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Runtime").strong());
        ui.label("Inference: tract (CPU, pure Rust)");
        ui.label("Silero VAD: loaded from models/silero_vad.onnx");
        ui.label("ECAPA-TDNN: loaded from models/ecapa_tdnn.onnx");
    });

    changed
}
