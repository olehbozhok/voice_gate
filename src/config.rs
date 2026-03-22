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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateMode {
    /// Open immediately on speech, close if verification fails.
    /// Lowest latency, brief leak possible for non-owner speech.
    Optimistic,
    /// Keep closed until speaker verification confirms the owner.
    /// Higher latency (~1.5s), no leak for non-owner speech.
    Strict,
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
