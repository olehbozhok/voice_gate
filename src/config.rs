//! Application configuration with serde persistence.
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Returns the platform-appropriate default directory for ONNX model files.
fn default_models_dir() -> PathBuf {
    if let Some(data_dir) = dirs::data_dir() {
        data_dir.join("voice-gate").join("models")
    } else {
        PathBuf::from("models")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub audio: AudioConfig,
    pub vad: VadConfig,
    pub speaker: SpeakerConfig,
    pub gate: GateConfig,
    pub profiles_dir: PathBuf,
    /// Directory where ONNX model files are stored.
    /// Default: `%APPDATA%/voice-gate/models/` (Windows) or `~/.local/share/voice-gate/models/`.
    #[serde(default = "default_models_dir")]
    pub models_dir: PathBuf,
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
pub struct VadConfig {
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerConfig {
    pub similarity_threshold: f32,
    pub min_enrollment_seconds: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate mode
// ─────────────────────────────────────────────────────────────────────────────

/// Gate operating mode — each variant implements a different tradeoff
/// between latency (clipping the owner) and leakage (passing other voices).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateMode {
    /// **"Open first, verify later"** (recommended).
    ///
    /// The gate opens instantly when VAD detects speech. Speaker verification
    /// runs in the background. If verification determines the speaker is NOT
    /// the owner, the gate closes.
    ///
    /// - Owner's voice: **never clipped**.
    /// - Other voices: may leak for ~0.5–1s until verification completes.
    /// - Best for: calls, meetings, streaming — where clipping yourself
    ///   is worse than briefly hearing someone else.
    #[default]
    Optimistic,

    /// **"Verify first, then open"** (strict).
    ///
    /// The gate stays closed until speaker verification confirms the owner.
    /// Audio is buffered and retroactively passed if verified.
    ///
    /// - Owner's voice: first ~0.5–1s **may be lost** (unless pre-buffered).
    /// - Other voices: never leak.
    /// - Best for: recording, security — where leaking others is unacceptable.
    Strict,

    /// **VAD-only gate** (no speaker verification).
    ///
    /// Opens when any speech is detected, closes on silence. Useful as a
    /// baseline or when no voice profile is enrolled.
    ///
    /// - All voices pass through.
    /// - Background noise and silence are gated.
    VadOnly,
}

impl GateMode {
    /// Evaluate the gate decision for a single audio frame.
    ///
    /// This is a **pure function** — all state is carried in [`GateInput`],
    /// making it trivial to test and reason about.
    pub fn evaluate(self, input: &GateInput) -> GateDecision {
        match self {
            GateMode::Optimistic => evaluate_optimistic(input),
            GateMode::Strict => evaluate_strict(input),
            GateMode::VadOnly => evaluate_vad_only(input),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate input / output
// ─────────────────────────────────────────────────────────────────────────────

/// All inputs needed for the gate decision, collected by the processor
/// and passed to [`GateMode::evaluate`] each frame.
pub struct GateInput {
    /// VAD speech probability for this frame (0.0–1.0).
    pub speech_probability: f32,
    /// VAD threshold — speech detected when `speech_probability >= vad_threshold`.
    pub vad_threshold: f32,
    /// At least one speaker verification has completed.
    pub verified: bool,
    /// Most recent speaker similarity score (0.0–1.0), if any.
    pub similarity: Option<f32>,
    /// Similarity threshold for "is owner" decision.
    pub similarity_threshold: f32,
    /// Whether there is an enrolled voice profile.
    pub has_profile: bool,
    /// How long the gate should stay open after speech stops (ms).
    pub hold_time_ms: u32,
    /// Duration of continuous silence in milliseconds.
    pub silence_ms: u32,
}

impl GateInput {
    /// Returns `true` if VAD considers the current frame to contain speech.
    fn is_speech(&self) -> bool {
        self.speech_probability >= self.vad_threshold
    }

    /// Returns `true` if the most recent verification says "owner".
    fn is_owner(&self) -> bool {
        match self.similarity {
            Some(sim) => sim >= self.similarity_threshold,
            None => false,
        }
    }

    /// Returns `true` if we're within the hold window after speech ended.
    fn in_hold_window(&self) -> bool {
        !self.is_speech() && self.silence_ms <= self.hold_time_ms
    }
}

/// Decision returned by [`GateMode::evaluate`] each frame.
pub struct GateDecision {
    /// Whether audio should pass through the gate.
    pub pass_audio: bool,
    /// Whether the verification buffer should be cleared.
    /// Used when the mode determines the current audio context is stale.
    pub flush_verification: bool,
}

impl GateDecision {
    /// Audio passes through, verification state preserved.
    fn pass() -> Self {
        Self {
            pass_audio: true,
            flush_verification: false,
        }
    }

    /// Audio blocked, verification state preserved.
    fn block() -> Self {
        Self {
            pass_audio: false,
            flush_verification: false,
        }
    }

    /// Audio blocked, verification state flushed (stale context).
    fn block_and_flush() -> Self {
        Self {
            pass_audio: false,
            flush_verification: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Optimistic: open immediately, revoke if verification fails.
///
/// | Speech? | Verified? | Owner? | Profile? | → Decision  |
/// |---------|-----------|--------|----------|-------------|
/// | no      | —         | —      | —        | hold/block  |
/// | yes     | no        | —      | no       | pass        |
/// | yes     | no        | —      | yes      | **pass** ←  |
/// | yes     | yes       | yes    | yes      | pass        |
/// | yes     | yes       | no     | yes      | **block**   |
fn evaluate_optimistic(input: &GateInput) -> GateDecision {
    // ── No speech ───────────────────────────────────────────────
    if !input.is_speech() {
        if input.in_hold_window() {
            return GateDecision::pass();
        }
        if input.silence_ms < 2000 {
            return GateDecision::block();
        }
        return GateDecision::block_and_flush();
    }

    // ── Speech detected ─────────────────────────────────────────

    if !input.has_profile {
        return GateDecision::pass();
    }

    if input.verified {
        return if input.is_owner() {
            GateDecision::pass()
        } else {
            GateDecision::block()
        };
    }

    // Verification hasn't completed yet → PASS (optimistic assumption).
    GateDecision::pass()
}

/// Strict: block until verification confirms the owner.
///
/// | Speech? | Verified? | Owner? | Profile? | → Decision  |
/// |---------|-----------|--------|----------|-------------|
/// | no      | —         | —      | —        | hold/block  |
/// | yes     | no        | —      | no       | pass        |
/// | yes     | no        | —      | yes      | **block**   |
/// | yes     | yes       | yes    | yes      | pass        |
/// | yes     | yes       | no     | yes      | block       |
fn evaluate_strict(input: &GateInput) -> GateDecision {
    if !input.is_speech() {
        if input.in_hold_window() {
            return GateDecision::pass();
        }
        return GateDecision::block_and_flush();
    }

    if !input.has_profile {
        return GateDecision::pass();
    }

    if input.verified && input.is_owner() {
        return GateDecision::pass();
    }

    GateDecision::block()
}

/// VAD-only: pass all speech, block silence.
fn evaluate_vad_only(input: &GateInput) -> GateDecision {
    if input.is_speech() {
        return GateDecision::pass();
    }

    if input.in_hold_window() {
        return GateDecision::pass();
    }

    GateDecision::block_and_flush()
}

// ─────────────────────────────────────────────────────────────────────────────
// Gate config
// ─────────────────────────────────────────────────────────────────────────────

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
                sample_rate: 16_000,
                channels: 1,
                frame_samples: 512,
                input_device: None,
                output_device: None,
            },
            vad: VadConfig { threshold: 0.5 },
            speaker: SpeakerConfig {
                similarity_threshold: 0.70,
                min_enrollment_seconds: 10.0,
            },
            gate: GateConfig {
                hold_time_ms: 300,
                pre_buffer_ms: 100,
                mode: GateMode::Optimistic,
            },
            profiles_dir: PathBuf::from("profiles"),
            models_dir: default_models_dir(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
                log::warn!("Failed to parse config ({}), defaults", e);
                Self::default()
            }),
            Err(_) => {
                log::info!("No config file, using defaults");
                Self::default()
            }
        }
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
