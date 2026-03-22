//! Application configuration with serde persistence.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub speaker: SpeakerConfig,
    pub gate: GateConfig,
    pub profiles_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_samples: usize,
    /// Selected input device name. `None` = system default.
    #[serde(default)]
    pub input_device: Option<String>,
    /// Selected output device name. `None` = system default.
    #[serde(default)]
    pub output_device: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig { pub threshold: f32 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConfig {
    pub similarity_threshold: f32,
    pub min_enrollment_seconds: f32,
}

/// How the gate decides when to open relative to speaker verification.
///
/// Each mode implements [`GateMode::pre_verification_decision`] which determines
/// what happens when speech is detected but the speaker hasn't been verified yet
/// (not enough audio accumulated, or verification is still running).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateMode {
    /// Open the gate immediately when speech is detected, before speaker
    /// verification completes. If verification later determines the speaker
    /// is not the owner, the gate closes.
    ///
    /// Trade-off: lowest latency (instant response), but a non-owner's first
    /// ~1.5s of speech leaks through until verification kicks in.
    Optimistic,

    /// Keep the gate closed until speaker verification positively confirms
    /// the owner. No audio passes until the model has enough data (~1.5s)
    /// and the embedding matches the enrolled profile.
    ///
    /// Trade-off: no leak for non-owner speech, but the owner experiences
    /// ~1.5s of silence at the start of each utterance after a long pause.
    Strict,
}

impl GateMode {
    /// Decide whether to open the gate when there is not yet enough audio
    /// for a full speaker verification.
    ///
    /// # Arguments
    /// * `verified_once` — whether at least one verification has completed
    ///   in the current speech segment.
    /// * `last_result` — the most recent verification result `(is_owner, similarity)`,
    ///   if any. Preserved across brief silences.
    ///
    /// # Returns
    /// `(is_owner, similarity)` to use for the gate decision this frame.
    pub fn pre_verification_decision(
        self,
        verified_once: bool,
        last_result: Option<(bool, f32)>,
    ) -> (bool, f32) {
        if verified_once {
            // Already verified at least once — reuse the last result
            // regardless of mode. This covers brief pauses and the gap
            // while the next verification window accumulates.
            return last_result.unwrap_or((false, 0.0));
        }

        match self {
            GateMode::Optimistic => (true, 1.0),
            GateMode::Strict => (false, 0.0),
        }
    }
}

impl Default for GateMode {
    fn default() -> Self { Self::Optimistic }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    pub hold_time_ms: u32,
    pub pre_buffer_ms: u32,
    /// Controls whether the gate opens before or after speaker verification.
    #[serde(default)]
    pub mode: GateMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig {
                sample_rate: 16_000, channels: 1, frame_samples: 512,
                input_device: None, output_device: None,
            },
            vad: VadConfig { threshold: 0.5 },
            speaker: SpeakerConfig { similarity_threshold: 0.70, min_enrollment_seconds: 10.0 },
            gate: GateConfig { hold_time_ms: 300, pre_buffer_ms: 100, mode: GateMode::Optimistic },
            profiles_dir: PathBuf::from("profiles"),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                log::warn!("Failed to parse config ({}), defaults", e); Self::default()
            }),
            Err(_) => { log::info!("No config file, using defaults"); Self::default() }
        }
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?; Ok(())
    }
    pub fn pre_buffer_samples(&self) -> usize {
        (self.audio.sample_rate * self.gate.pre_buffer_ms / 1000) as usize
    }
}
