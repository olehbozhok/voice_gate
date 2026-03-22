//! Enrollment wizard — guides user through voice recording + profile list.

use egui::{Color32, RichText, Ui, Vec2};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::speaker::enrollment::EnrollmentState;
use crate::speaker::profile::ProfileStore;

const PROMPTS: &[&str] = &[
    "Read: \"The quick brown fox jumps over the lazy dog.\"",
    "Count from one to twenty naturally.",
    "Describe what you see around you.",
    "Tell about your favourite movie.",
    "Read: \"She sells seashells by the seashore.\"",
];

/// Persistent UI state for the enrollment view (not serialized).
#[derive(Default)]
pub struct EnrollmentViewState {
    /// Maps profile index → edit buffer. Present = editing mode.
    editing: HashMap<usize, String>,
}

#[allow(clippy::too_many_arguments)]
pub fn show(
    ui: &mut Ui,
    state: &EnrollmentState,
    speech_secs: f32,
    min_secs: f32,
    profile_store: &Arc<RwLock<ProfileStore>>,
    view_state: &mut EnrollmentViewState,
    on_start: &mut dyn FnMut(),
    on_finalize: &mut dyn FnMut(),
    on_cancel: &mut dyn FnMut(),
) {
    ui.heading("Voice Enrollment");
    ui.add_space(4.0);

    // ── Profile list ────────────────────────────────────────────────
    show_profile_list(ui, profile_store, view_state);

    ui.add_space(12.0);
    ui.separator();
    ui.add_space(8.0);

    // ── Enrollment state machine ────────────────────────────────────
    match state {
        EnrollmentState::Idle => {
            ui.label("Record a new voice profile.");
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.label(RichText::new("Tips:").strong());
                ui.label("Speak at normal volume and vary your intonation.");
                ui.label(format!("You need at least {:.0}s of speech.", min_secs));
            });
            ui.add_space(12.0);
            if ui
                .button(RichText::new("Start Recording").size(16.0))
                .clicked()
            {
                on_start();
            }
        }
        EnrollmentState::Recording {
            speech_seconds: secs,
        } => {
            let progress = (secs / min_secs).clamp(0.0, 1.0);
            let bar_color = if progress >= 1.0 {
                Color32::from_rgb(50, 205, 50)
            } else {
                Color32::from_rgb(70, 130, 255)
            };

            ui.label(
                RichText::new(format!("Recording — {:.1}s / {:.0}s", secs, min_secs))
                    .size(16.0)
                    .color(Color32::from_rgb(220, 60, 60)),
            );
            ui.add_space(4.0);

            let (rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 16.0), egui::Sense::hover());
            ui.painter()
                .rect_filled(rect, 4.0, Color32::from_rgb(40, 40, 45));
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
                if ui
                    .add_enabled(can_finish, egui::Button::new("Finish & Save"))
                    .clicked()
                {
                    on_finalize();
                }
                if ui.button("Cancel").clicked() {
                    on_cancel();
                }
            });
        }
        EnrollmentState::Processing => {
            ui.spinner();
            ui.label("Processing...");
        }
        EnrollmentState::Done => {
            ui.label(
                RichText::new("Enrollment complete! Profile saved.")
                    .size(16.0)
                    .color(Color32::from_rgb(50, 205, 50)),
            );
            ui.add_space(8.0);
            if ui.button("Add Another").clicked() {
                on_cancel();
            }
        }
        EnrollmentState::Failed(reason) => {
            ui.label(
                RichText::new(format!("Failed: {}", reason)).color(Color32::from_rgb(220, 60, 60)),
            );
            ui.add_space(8.0);
            if ui.button("Try Again").clicked() {
                on_cancel();
            }
        }
    }
}

/// Show the list of enrolled profiles with inline rename and delete.
fn show_profile_list(
    ui: &mut Ui,
    store: &Arc<RwLock<ProfileStore>>,
    view_state: &mut EnrollmentViewState,
) {
    let profiles = store.read().profiles().to_vec();

    if profiles.is_empty() {
        ui.label(RichText::new("No voice profiles enrolled.").weak());
        return;
    }

    ui.label(RichText::new(format!("Enrolled profiles ({})", profiles.len())).strong());
    ui.add_space(4.0);

    let mut rename: Option<(usize, String)> = None;
    let mut delete: Option<usize> = None;

    for (i, profile) in profiles.iter().enumerate() {
        ui.horizontal(|ui| {
            if let Some(buf) = view_state.editing.get_mut(&i) {
                // ── Editing mode ────────────────────────────────
                let response = ui.text_edit_singleline(buf);

                if response.lost_focus() || ui.input(|inp| inp.key_pressed(egui::Key::Enter)) {
                    let new_name = buf.trim().to_string();
                    if !new_name.is_empty() {
                        rename = Some((i, new_name));
                    }
                    view_state.editing.remove(&i);
                }

                if ui.small_button("✕").on_hover_text("Cancel edit").clicked() {
                    view_state.editing.remove(&i);
                }

                // Auto-focus the text field when it first appears.
                response.request_focus();
            } else {
                // ── Display mode ────────────────────────────────
                ui.label(&profile.name);
                ui.label(RichText::new(&profile.created_at).weak().small());

                if ui.small_button("✏").on_hover_text("Rename").clicked() {
                    view_state.editing.insert(i, profile.name.clone());
                }

                if ui.small_button("🗑").on_hover_text("Delete").clicked() {
                    delete = Some(i);
                }
            }
        });
    }

    // Apply actions outside the profile read borrow.
    if let Some((idx, new_name)) = rename {
        if let Err(e) = store.write().rename(idx, new_name) {
            log::error!("Failed to rename profile: {}", e);
        }
    }
    if let Some(idx) = delete {
        if let Err(e) = store.write().delete(idx) {
            log::error!("Failed to delete profile: {}", e);
        }
        // Clean up any editing state for deleted/shifted indices.
        view_state.editing.clear();
    }
}
