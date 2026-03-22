//! Main dashboard — level meters, gate state indicator, start/stop.

use crate::config::Config;
use crate::pipeline::processor::PipelineTelemetry;
use crate::pipeline::state_machine::GateState;
use egui::{Color32, RichText, Ui, Vec2};
use parking_lot::RwLock;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut Ui,
    telemetry: &Arc<RwLock<PipelineTelemetry>>,
    config: &Arc<RwLock<Config>>,
    is_running: bool,
    has_profile: bool,
    on_toggle: &mut dyn FnMut(),
    is_recording: bool,
    on_record_toggle: &mut dyn FnMut(),
) {
    let t = telemetry.read().clone();
    let cfg = config.read();

    ui.heading("Voice Gate");
    ui.add_space(8.0);

    // Gate state indicator
    let (color, label) = match t.gate_state {
        GateState::Silent => (Color32::from_rgb(100, 100, 100), "SILENT"),
        GateState::MyVoice => (Color32::from_rgb(50, 205, 50), "MY VOICE"),
        GateState::OtherVoice => (Color32::from_rgb(220, 50, 50), "OTHER VOICE"),
        GateState::Trailing => (Color32::from_rgb(255, 200, 50), "TRAILING"),
    };
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), egui::Sense::hover());
        ui.painter().circle_filled(rect.center(), 8.0, color);
        ui.label(RichText::new(label).size(18.0).strong().color(color));
    });

    ui.add_space(12.0);

    // Level meters
    ui.group(|ui| {
        ui.label("Input Level");
        level_bar(
            ui,
            t.input_level.clamp(0.0, 1.0),
            Color32::from_rgb(70, 130, 255),
        );
        ui.add_space(4.0);
        ui.label("VAD Speech Probability");
        level_bar(ui, t.vad_probability, Color32::from_rgb(50, 205, 50));
        ui.add_space(4.0);
        ui.label("Speaker Similarity");
        level_bar(
            ui,
            t.speaker_similarity.clamp(0.0, 1.0),
            similarity_color(t.speaker_similarity),
        );
    });

    ui.add_space(12.0);

    // Controls
    ui.horizontal(|ui| {
        let btn = if is_running { "Stop" } else { "Start" };
        if ui.button(RichText::new(btn).size(16.0)).clicked() {
            on_toggle();
        }

        if is_running {
            let rec_btn = if is_recording {
                "Stop Recording"
            } else {
                "Record Test"
            };
            let rec_color = if is_recording {
                Color32::from_rgb(220, 60, 60)
            } else {
                Color32::GRAY
            };
            if ui
                .button(RichText::new(rec_btn).size(14.0).color(rec_color))
                .clicked()
            {
                on_record_toggle();
            }
        }

        if !has_profile {
            ui.label(
                RichText::new("No voice profile — all speech passes through")
                    .color(Color32::from_rgb(255, 180, 50)),
            );
        }
    });

    ui.add_space(8.0);
    ui.label(RichText::new("Inference: tract (CPU)").weak().small());

    // Details
    egui::CollapsingHeader::new("Details").show(ui, |ui| {
        let vad_above = t.vad_probability >= cfg.vad.threshold;
        let sim_above = t.speaker_similarity >= cfg.speaker.similarity_threshold;

        egui::Grid::new("telem")
            .num_columns(2)
            .spacing([20.0, 4.0])
            .show(ui, |ui| {
                ui.label("Gate:");
                ui.label(format!("{}", t.gate_state));
                ui.end_row();

                ui.label("RMS:");
                ui.label(format!("{:.4}", t.input_level));
                ui.end_row();

                ui.label("VAD:");
                let vad_color = if vad_above {
                    Color32::from_rgb(50, 205, 50)
                } else {
                    Color32::from_rgb(180, 180, 180)
                };
                ui.label(
                    RichText::new(format!(
                        "{:.3} (thr: {:.2})",
                        t.vad_probability, cfg.vad.threshold
                    ))
                    .color(vad_color),
                );
                ui.end_row();

                ui.label("Similarity:");
                let sim_color = if sim_above {
                    Color32::from_rgb(50, 205, 50)
                } else {
                    Color32::from_rgb(230, 160, 60)
                };
                ui.label(
                    RichText::new(format!(
                        "{:.3} (thr: {:.2})",
                        t.speaker_similarity, cfg.speaker.similarity_threshold
                    ))
                    .color(sim_color),
                );
                ui.end_row();

                ui.label("Open:");
                let open_color = if t.gate_open {
                    Color32::from_rgb(50, 205, 50)
                } else {
                    Color32::from_rgb(180, 180, 180)
                };
                ui.label(RichText::new(if t.gate_open { "Yes" } else { "No" }).color(open_color));
                ui.end_row();
            });
    });
}

fn level_bar(ui: &mut Ui, value: f32, fill: Color32) {
    let size = Vec2::new(ui.available_width(), 14.0);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    ui.painter()
        .rect_filled(rect, 4.0, Color32::from_rgb(40, 40, 45));
    let mut fr = rect;
    fr.set_right(rect.left() + rect.width() * value.clamp(0.0, 1.0));
    ui.painter().rect_filled(fr, 4.0, fill);
}

fn similarity_color(sim: f32) -> Color32 {
    let c = sim.clamp(0.0, 1.0);
    Color32::from_rgb((220.0 * (1.0 - c)) as u8, (205.0 * c) as u8, 50)
}
