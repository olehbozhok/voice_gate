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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig { pub threshold: f32 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConfig {
    pub similarity_threshold: f32,
    pub min_enrollment_seconds: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    pub hold_time_ms: u32,
    pub pre_buffer_ms: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio: AudioConfig { sample_rate: 16_000, channels: 1, frame_samples: 512 },
            vad: VadConfig { threshold: 0.5 },
            speaker: SpeakerConfig { similarity_threshold: 0.70, min_enrollment_seconds: 10.0 },
            gate: GateConfig { hold_time_ms: 300, pre_buffer_ms: 100 },
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
