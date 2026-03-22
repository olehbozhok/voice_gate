//! Unified error types.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("no input device available")]
    NoInputDevice,
    #[error("no output device available")]
    NoOutputDevice,
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("stream error: {0}")]
    Stream(String),
    #[error("device enumeration failed: {0}")]
    DeviceEnumeration(String),
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("profile not found at '{0}'")]
    NotFound(String),
    #[error("failed to save profile: {0}")]
    SaveFailed(String),
    #[error("failed to parse profile: {0}")]
    ParseFailed(String),
    #[error("enrollment too short (need >= {min_seconds}s, got {got_seconds}s)")]
    TooShort { min_seconds: f32, got_seconds: f32 },
}
