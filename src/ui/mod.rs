//! GUI layer (egui/eframe).
pub mod main_view;
pub mod enrollment_view;
pub mod settings_view;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveView { Main, Enrollment, Settings }
