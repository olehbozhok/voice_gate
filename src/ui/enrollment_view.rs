//! Enrollment wizard — guides user through voice recording.

use egui::{Color32, RichText, Ui, Vec2};
use crate::speaker::enrollment::EnrollmentState;

const PROMPTS: &[&str] = &[
    "Read: \"The quick brown fox jumps over the lazy dog.\"",
    "Count from one to twenty naturally.",
    "Describe what you see around you.",
    "Tell about your favourite movie.",
    "Read: \"She sells seashells by the seashore.\"",
];

pub fn show(
    ui: &mut Ui, state: &EnrollmentState, speech_secs: f32, min_secs: f32,
    on_start: &mut dyn FnMut(), on_finalize: &mut dyn FnMut(), on_cancel: &mut dyn FnMut(),
) {
    ui.heading("Voice Enrollment");
    ui.add_space(4.0);
    ui.label("Record your voice so Voice Gate can recognise you.");
    ui.add_space(12.0);

    match state {
        EnrollmentState::Idle => {
            ui.label("Click Start and speak naturally for at least 10 seconds.");
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label(RichText::new("Tips:").strong());
                ui.label("Speak at normal volume and vary your intonation.");
                ui.label(format!("You need at least {:.0}s of speech.", min_secs));
            });
            ui.add_space(12.0);
            if ui.button(RichText::new("Start Recording").size(16.0)).clicked() { on_start(); }
        }
        EnrollmentState::Recording { speech_seconds: secs } => {
            let progress = (secs / min_secs).clamp(0.0, 1.0);
            let bar_color = if progress >= 1.0 { Color32::from_rgb(50, 205, 50) } else { Color32::from_rgb(70, 130, 255) };

            ui.label(RichText::new(format!("Recording — {:.1}s / {:.0}s", secs, min_secs))
                .size(16.0).color(Color32::from_rgb(220, 60, 60)));
            ui.add_space(4.0);

            let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 16.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 4.0, Color32::from_rgb(40, 40, 45));
            let mut fill = rect;
            fill.set_right(rect.left() + rect.width() * progress);
            ui.painter().rect_filled(fill, 4.0, bar_color);

            ui.add_space(12.0);
            let idx = (speech_secs / 5.0) as usize % PROMPTS.len();
            ui.group(|ui| {
                ui.label(RichText::new("Say:").strong());
                ui.label(RichText::new(PROMPTS[idx]).italics());
            });

            ui.add_space(12.0);
            ui.horizontal(|ui| {
                let can_finish = *secs >= min_secs;
                if ui.add_enabled(can_finish, egui::Button::new("Finish & Save")).clicked() { on_finalize(); }
                if ui.button("Cancel").clicked() { on_cancel(); }
            });
        }
        EnrollmentState::Processing => { ui.spinner(); ui.label("Processing..."); }
        EnrollmentState::Done => {
            ui.label(RichText::new("Enrollment complete! Profile saved.")
                .size(16.0).color(Color32::from_rgb(50, 205, 50)));
        }
        EnrollmentState::Failed(reason) => {
            ui.label(RichText::new(format!("Failed: {}", reason)).color(Color32::from_rgb(220, 60, 60)));
            ui.add_space(8.0);
            if ui.button("Try Again").clicked() { on_cancel(); }
        }
    }
}
