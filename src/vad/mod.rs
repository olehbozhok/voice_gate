//! Voice Activity Detection.
pub mod silero;

#[derive(Debug, Clone, Copy)]
pub struct VadResult {
    pub speech_probability: f32,
    pub is_speech: bool,
}
