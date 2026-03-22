//! GUI layer (egui/eframe).
pub mod enrollment_view;
pub mod main_view;
pub mod model_setup_view;
pub mod settings_view;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveView {
    Main,
    Enrollment,
    Settings,
}
