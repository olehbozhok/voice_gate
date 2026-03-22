//! Unified error types.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("no input device available")]
    NoInputDevice,
    #[error("no output device available")]
    NoOutputDevice,
}

#[derive(Debug, Error)]
pub enum ProfileError {
    #[error("profile not found at '{0}'")]
    NotFound(String),
    #[error("failed to save profile: {0}")]
    SaveFailed(String),
    #[error("failed to parse profile: {0}")]
    ParseFailed(String),
}
