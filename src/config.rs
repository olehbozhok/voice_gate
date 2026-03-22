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

/// Snapshot of the current speaker verification state, passed to
/// [`GateMode::should_open`] so each mode can make its own decision.
pub struct VerificationState {
    /// VAD detected speech in this frame.
    pub is_speech: bool,
    /// At least one full verification has completed in this speech segment.
    pub verified: bool,
    /// Most recent speaker similarity score (0.0–1.0), if any.
    pub similarity: Option<f32>,
    /// Configured similarity threshold for "is owner" decision.
    pub threshold: f32,
    /// Whether there is an enrolled voice profile.
    pub has_profile: bool,
}

impl GateMode {
    /// Decide whether the gate should be open for the current frame.
    ///
    /// Each variant implements its own algorithm:
    ///
    /// - **Optimistic**: open on speech immediately; close only after
    ///   verification fails. Prioritises low latency.
    /// - **Strict**: stay closed until verification positively confirms
    ///   the owner. Prioritises security.
    pub fn should_open(self, state: &VerificationState) -> bool {
        if !state.is_speech {
            return false;
        }

        if !state.has_profile {
            // No profile enrolled — let all speech through.
            return true;
        }

        match self {
            GateMode::Optimistic => {
                // Open by default. Close only when verification has
                // explicitly determined the speaker is NOT the owner.
                match (state.verified, state.similarity) {
                    (true, Some(sim)) => sim >= state.threshold,
                    _ => true, // not yet verified — trust
                }
            }
            GateMode::Strict => {
                // Closed by default. Open only when verification has
                // explicitly confirmed the speaker IS the owner.
                match (state.verified, state.similarity) {
                    (true, Some(sim)) => sim >= state.threshold,
                    _ => false, // not yet verified — block
                }
            }
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
