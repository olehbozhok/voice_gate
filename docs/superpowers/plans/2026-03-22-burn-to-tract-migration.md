# Burn to Tract Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Burn with tract for ONNX inference, removing build-time code generation and the `B: Backend` generic parameter.

**Architecture:** Runtime ONNX model loading via tract-onnx, hidden behind `OnnxModel` wrapper in `src/inference.rs`. Models loaded at app startup, no build.rs. All Burn types, feature flags, and generic parameters removed.

**Tech Stack:** tract-onnx 0.22 (pure Rust ONNX inference), existing cpal/egui/eframe stack unchanged.

**Spec:** `docs/superpowers/specs/2026-03-22-burn-to-tract-migration-design.md`

---

## File Structure

### Create
| File | Responsibility |
|------|---------------|
| `src/inference.rs` | `OnnxModel` wrapper — hides tract internals, provides `Input`/`Output`/`ModelState` types |

### Delete
| File | Reason |
|------|--------|
| `build.rs` | No build-time ONNX compilation |
| `src/backend.rs` | No `AppBackend` type alias |
| `src/model/mod.rs` | No generated model includes |

### Modify
| File | Changes |
|------|---------|
| `Cargo.toml` | Remove burn/burn-import, add tract-onnx, remove feature flags |
| `src/main.rs` | Remove `mod model`/`mod backend`, add `mod inference`, update log line |
| `src/vad/silero.rs` | Rewrite: use `OnnxModel` + `Input`/`Output`/`ModelState` |
| `src/speaker/embedding.rs` | Rewrite: use `OnnxModel` + `Input`/`Output` |
| `src/pipeline/processor.rs` | Remove `B: Backend` generic |
| `src/app.rs` | Remove `AppBackend`, pass model paths |
| `src/ui/main_view.rs` | Remove `backend_name()` call |
| `src/ui/settings_view.rs` | Remove `backend_name()` + cfg conditionals |

---

## Task 1: Update Cargo.toml

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Replace dependencies and remove feature flags**

Replace the full `Cargo.toml` content with burn and feature sections removed:

```toml
[package]
name = "voice-gate"
version = "0.2.0"
edition = "2021"
authors = ["Your Name <your@email.com>"]
description = """
Voice Gate — intelligent microphone gate that activates only for *your* voice.
Silero VAD for voice detection + ECAPA-TDNN for speaker verification.
"""
license = "MIT"
readme = "README.md"

[dependencies]
# ── ONNX Inference (pure Rust) ─────────────────────────────────────────
tract-onnx = "0.22"

# ── Audio I/O ──────────────────────────────────────────────────────────
cpal = "0.15"
hound = "3.5"

# ── GUI ──────────────────────────────────────────────────────────────
eframe = "0.31"
egui = "0.31"

# ── Utilities ──────────────────────────────────────────────────────────
anyhow = "1"
thiserror = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
log = "0.4"
env_logger = "0.11"
crossbeam-channel = "0.5"
parking_lot = "0.12"

[profile.release]
opt-level = 3
lto = "thin"
```

Key changes:
- Removed `burn` dependency
- Removed `[features]` section entirely (cpu/wgpu/cuda/dev)
- Removed `[build-dependencies]` section (burn-import)
- Added `tract-onnx = "0.22"`

- [ ] **Step 2: Verify it parses**

Run: `cargo check 2>&1 | head -5`

Expected: compilation errors about missing modules (not Cargo.toml parse errors). This confirms the manifest is valid.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "chore: replace burn with tract-onnx in Cargo.toml"
```

---

## Task 2: Delete obsolete files

**Files:**
- Delete: `build.rs`
- Delete: `src/backend.rs`
- Delete: `src/model/mod.rs`

- [ ] **Step 1: Delete the three files**

```bash
rm build.rs src/backend.rs src/model/mod.rs
rmdir src/model 2>/dev/null || true
```

- [ ] **Step 2: Update src/main.rs — remove old modules, add inference**

Replace the full content of `src/main.rs` with:

```rust
//! Voice Gate — entry point.

mod app;
mod audio;
mod config;
mod error;
mod inference;
mod pipeline;
mod speaker;
mod ui;
mod vad;

fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    log::info!("Voice Gate v{}", env!("CARGO_PKG_VERSION"));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Voice Gate")
            .with_inner_size([520.0, 560.0])
            .with_min_inner_size([400.0, 400.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Voice Gate",
        options,
        Box::new(|cc| Ok(Box::new(app::VoiceGateApp::new(cc)))),
    )
}
```

Changes: removed `mod backend`, `mod model`, added `mod inference`, removed `backend::backend_name()` log line.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: delete build.rs, backend.rs, model/mod.rs; update main.rs"
```

---

## Task 3: Create src/inference.rs (OnnxModel wrapper)

**Files:**
- Create: `src/inference.rs`

- [ ] **Step 1: Write the inference wrapper**

Create `src/inference.rs` with full content:

```rust
//! ONNX inference wrapper — hides the tract runtime behind a stable API.
//!
//! To swap the inference backend (e.g. to ort), change only this file.
//! No other module imports tract types directly.

use std::path::Path;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

/// Opaque wrapper over an optimized ONNX model.
pub struct OnnxModel {
    plan: TypedRunnableModel<TypedFact>,
}

/// Opaque handle for stateful model data (e.g. LSTM hidden/cell state).
/// Passed back into subsequent inference calls without copying.
pub struct ModelState {
    inner: TValue,
}

/// Typed model input — no framework-specific types exposed.
pub enum Input {
    /// Dense f32 tensor with explicit shape.
    F32 { shape: Vec<usize>, data: Vec<f32> },
    /// Dense i64 tensor with explicit shape.
    I64 { shape: Vec<usize>, data: Vec<i64> },
    /// Opaque state from a previous inference call.
    State(ModelState),
}

/// Opaque model output with typed accessors.
pub struct Output {
    inner: TValue,
}

// ── OnnxModel ────────────────────────────────────────────────────────────

impl OnnxModel {
    /// Load an ONNX model from disk, optimize the graph, and prepare for
    /// inference.
    pub fn load(path: &Path) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("failed to load ONNX model: {}", path.display()))?
            .into_optimized()
            .with_context(|| format!("failed to optimize model: {}", path.display()))?
            .into_runnable()
            .with_context(|| format!("failed to prepare model: {}", path.display()))?;
        Ok(Self { plan })
    }

    /// Run inference with the given inputs.
    pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Output>> {
        let tv: TVec<TValue> = inputs
            .into_iter()
            .map(|inp| inp.into_tvalue())
            .collect::<Result<_>>()?;
        let outputs = self.plan.run(tv)?;
        Ok(outputs.into_iter().map(|v| Output { inner: v }).collect())
    }
}

// ── Input ────────────────────────────────────────────────────────────────

impl Input {
    fn into_tvalue(self) -> Result<TValue> {
        match self {
            Input::F32 { shape, data } => {
                let tensor = tract_ndarray::Array::from_shape_vec(
                    tract_ndarray::IxDyn(&shape),
                    data,
                )?;
                Ok(Tensor::from(tensor).into())
            }
            Input::I64 { shape, data } => {
                let tensor = tract_ndarray::Array::from_shape_vec(
                    tract_ndarray::IxDyn(&shape),
                    data,
                )?;
                Ok(Tensor::from(tensor).into())
            }
            Input::State(state) => Ok(state.inner),
        }
    }
}

// ── Output ───────────────────────────────────────────────────────────────

impl Output {
    /// Extract a single f32 scalar from the output.
    pub fn to_scalar_f32(&self) -> Result<f32> {
        Ok(*self.inner.to_scalar::<f32>()?)
    }

    /// Extract the output as a flat Vec<f32>.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(self.inner.as_slice::<f32>()?.to_vec())
    }

    /// Convert this output into an opaque ModelState for round-tripping.
    pub fn into_state(self) -> ModelState {
        ModelState { inner: self.inner }
    }
}

// ── ModelState ───────────────────────────────────────────────────────────

impl ModelState {
    /// Create a zero-filled f32 state tensor with the given shape.
    pub fn zeros_f32(shape: &[usize]) -> Self {
        let tensor = tract_ndarray::Array::<f32, _>::zeros(
            tract_ndarray::IxDyn(shape),
        );
        Self {
            inner: Tensor::from(tensor).into(),
        }
    }
}
```

- [ ] **Step 2: Verify it compiles in isolation**

Run: `cargo check 2>&1 | grep "inference" | head -5`

Expected: errors from other modules (silero.rs, embedding.rs, etc.) that still reference burn — but `inference.rs` itself should not have errors. If there are errors in inference.rs, fix them first.

- [ ] **Step 3: Commit**

```bash
git add src/inference.rs
git commit -m "feat: add OnnxModel inference wrapper (hides tract internals)"
```

---

## Task 4: Rewrite src/vad/silero.rs

**Files:**
- Modify: `src/vad/silero.rs`

- [ ] **Step 1: Rewrite silero.rs**

Replace the full content of `src/vad/silero.rs` with:

```rust
//! Silero VAD — voice activity detection via tract ONNX runtime.
//!
//! The ONNX model is loaded at runtime from `models/silero_vad.onnx`.

use std::path::Path;

use anyhow::Result;

use crate::inference::{Input, ModelState, OnnxModel};
use super::VadResult;

/// Silero VAD wrapper — stateful LSTM model.
pub struct SileroVad {
    threshold: f32,
    model: OnnxModel,
    /// LSTM hidden state [2, 1, 64], carried across frames.
    h: Option<ModelState>,
    /// LSTM cell state [2, 1, 64], carried across frames.
    c: Option<ModelState>,
}

impl SileroVad {
    /// Load Silero VAD from an ONNX file.
    pub fn new(threshold: f32, model_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)?;
        log::info!("Silero VAD loaded from {}", model_path.display());

        let mut vad = Self {
            threshold,
            model,
            h: None,
            c: None,
        };
        vad.reset();

        // Warmup: run a dummy frame to pre-allocate buffers.
        let dummy = vec![0.0f32; 512];
        let _ = vad.process(&dummy);
        vad.reset();

        Ok(vad)
    }

    /// Run VAD inference on a single audio frame (typically 512 samples at 16 kHz).
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult> {
        let h = self.h.take().unwrap_or_else(|| ModelState::zeros_f32(&[2, 1, 64]));
        let c = self.c.take().unwrap_or_else(|| ModelState::zeros_f32(&[2, 1, 64]));

        let mut outputs = self.model.run(vec![
            Input::F32 { shape: vec![1, samples.len()], data: samples.to_vec() },
            Input::I64 { shape: vec![1], data: vec![16000] },
            Input::State(h),
            Input::State(c),
        ])?;

        // Extract in reverse order to avoid index shifts when removing.
        let c_out = outputs.remove(2).into_state();
        let h_out = outputs.remove(1).into_state();
        let prob = outputs.remove(0).to_scalar_f32()?;
        self.h = Some(h_out);
        self.c = Some(c_out);

        Ok(VadResult {
            speech_probability: prob,
            is_speech: prob >= self.threshold,
        })
    }

    /// Reset LSTM state to zeros.
    pub fn reset(&mut self) {
        self.h = Some(ModelState::zeros_f32(&[2, 1, 64]));
        self.c = Some(ModelState::zeros_f32(&[2, 1, 64]));
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}
```

Key changes from the old version:
- No generic `B: Backend` parameter
- No `#[cfg(has_silero_vad)]` conditionals or energy fallback
- Uses `OnnxModel`, `Input`, `ModelState` from `crate::inference`
- LSTM state stored as `Option<ModelState>` (using `take()` for zero-copy)
- Constructor takes `model_path: &Path` instead of `device: &B::Device`
- Warmup dummy inference in constructor

- [ ] **Step 2: Commit**

```bash
git add src/vad/silero.rs
git commit -m "feat: rewrite SileroVad to use OnnxModel wrapper"
```

---

## Task 5: Rewrite src/speaker/embedding.rs

**Files:**
- Modify: `src/speaker/embedding.rs`

- [ ] **Step 1: Rewrite embedding.rs**

Replace the full content of `src/speaker/embedding.rs` with:

```rust
//! ECAPA-TDNN speaker embedding extractor via tract ONNX runtime.
//!
//! The ONNX model is loaded at runtime from `models/ecapa_tdnn.onnx`.

use std::path::Path;

use anyhow::Result;

use crate::inference::{Input, OnnxModel};

/// Wrapper around the ECAPA-TDNN ONNX model.
pub struct EcapaTdnn {
    model: OnnxModel,
}

impl EcapaTdnn {
    /// Load ECAPA-TDNN from an ONNX file.
    pub fn new(model_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)?;
        log::info!("ECAPA-TDNN loaded from {}", model_path.display());

        let ecapa = Self { model };

        // Warmup: run a dummy waveform to pre-allocate buffers.
        let dummy = vec![0.0f32; 16000];
        let _ = ecapa.extract(&dummy);

        Ok(ecapa)
    }

    /// Extract a speaker embedding from an audio waveform.
    ///
    /// # Arguments
    ///
    /// * `waveform` — Mono f32 audio at 16 kHz. Ideally 2-5 seconds.
    ///
    /// # Returns
    ///
    /// An L2-normalized 192-dimensional embedding vector.
    pub fn extract(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        if waveform.len() < 16000 {
            log::warn!(
                "Waveform too short ({} samples = {:.1}s), embedding may be unreliable",
                waveform.len(),
                waveform.len() as f32 / 16000.0,
            );
        }

        let outputs = self.model.run(vec![
            Input::F32 {
                shape: vec![1, waveform.len()],
                data: waveform.to_vec(),
            },
        ])?;

        let embedding = outputs[0].to_f32_vec()?;
        Ok(l2_normalize(&embedding))
    }
}

/// L2-normalize a vector to unit length.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}
```

Key changes:
- No generic `B: Backend` parameter
- No `#[cfg(has_ecapa_tdnn)]` conditionals or zero-embedding fallback
- Uses `OnnxModel` and `Input` from `crate::inference`
- Constructor takes `model_path: &Path` instead of `device: &B::Device`
- Removed `is_available()` method (always available if loaded)

- [ ] **Step 2: Commit**

```bash
git add src/speaker/embedding.rs
git commit -m "feat: rewrite EcapaTdnn to use OnnxModel wrapper"
```

---

## Task 6: Update src/pipeline/processor.rs

**Files:**
- Modify: `src/pipeline/processor.rs`

- [ ] **Step 1: Remove Backend generic**

Replace the full content of `src/pipeline/processor.rs` with:

```rust
//! Audio processor — orchestrates VAD -> Speaker Verification -> Gate.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::speaker::cosine_similarity;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::VoiceProfile;
use crate::vad::silero::SileroVad;
use super::state_machine::{GateState, GateStateMachine};

/// Telemetry snapshot shared with the UI thread.
#[derive(Debug, Clone)]
pub struct PipelineTelemetry {
    pub gate_state: GateState,
    pub input_level: f32,
    pub vad_probability: f32,
    pub speaker_similarity: f32,
    pub gate_open: bool,
}

impl Default for PipelineTelemetry {
    fn default() -> Self {
        Self { gate_state: GateState::Silent, input_level: 0.0, vad_probability: 0.0, speaker_similarity: 0.0, gate_open: false }
    }
}

/// Main audio processor.
pub struct Processor {
    config: Config,
    vad: SileroVad,
    ecapa: EcapaTdnn,
    profile: Option<VoiceProfile>,
    gate: GateStateMachine,
    verification_buffer: VecDeque<f32>,
    verification_window_samples: usize,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
}

impl Processor {
    pub fn new(
        config: Config,
        vad: SileroVad,
        ecapa: EcapaTdnn,
        profile: Option<VoiceProfile>,
        telemetry: Arc<RwLock<PipelineTelemetry>>,
    ) -> Self {
        let verification_window_samples = (1.5 * config.audio.sample_rate as f32) as usize;
        Self {
            gate: GateStateMachine::new(config.gate.hold_time_ms),
            config, vad, ecapa, profile,
            verification_buffer: VecDeque::with_capacity(verification_window_samples),
            verification_window_samples,
            telemetry,
        }
    }

    /// Run the processing loop. Blocks until `rx_input` is closed.
    pub fn run(
        &mut self,
        rx_input: Receiver<Vec<f32>>,
        tx_output: Sender<Vec<f32>>,
    ) -> Result<()> {
        log::info!("Processor started (tract, CPU)");
        while let Ok(frame) = rx_input.recv() {
            let output = self.process_frame(&frame)?;
            let _ = tx_output.try_send(output);
        }
        log::info!("Processor stopping");
        Ok(())
    }

    fn process_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let input_level = crate::audio::rms(frame);

        // Stage 1: VAD
        let vad_result = self.vad.process(frame)?;

        // Stage 2: Speaker verification
        let (is_owner, similarity) = if vad_result.is_speech {
            self.run_speaker_verification(frame)
        } else {
            self.verification_buffer.clear();
            (false, 0.0)
        };

        // Stage 3: Gate
        let state = self.gate.update(vad_result.is_speech, is_owner);

        // Update telemetry for UI
        {
            let mut t = self.telemetry.write();
            t.gate_state = state;
            t.input_level = input_level;
            t.vad_probability = vad_result.speech_probability;
            t.speaker_similarity = similarity;
            t.gate_open = state.is_open();
        }

        if state.is_open() {
            Ok(frame.to_vec())
        } else {
            Ok(vec![0.0; frame.len()])
        }
    }

    fn run_speaker_verification(&mut self, frame: &[f32]) -> (bool, f32) {
        self.verification_buffer.extend(frame.iter());

        if self.verification_buffer.len() < self.verification_window_samples {
            let prev_sim = self.telemetry.read().speaker_similarity;
            let threshold = self.config.speaker.similarity_threshold;
            return (prev_sim >= threshold, prev_sim);
        }

        let window: Vec<f32> = self.verification_buffer.iter().copied().collect();
        let drain_count = self.verification_window_samples / 2;
        self.verification_buffer.drain(..drain_count);

        let profile = match &self.profile {
            Some(p) => p,
            None => return (true, 1.0),
        };

        match self.ecapa.extract(&window) {
            Ok(embedding) => {
                let sim = cosine_similarity(&profile.centroid, &embedding);
                let is_owner = sim >= self.config.speaker.similarity_threshold;
                log::trace!("Speaker similarity: {:.3} (owner: {})", sim, is_owner);
                (is_owner, sim)
            }
            Err(e) => {
                log::warn!("Embedding extraction failed: {}", e);
                (false, 0.0)
            }
        }
    }

    pub fn set_profile(&mut self, profile: VoiceProfile) {
        log::info!("Profile updated: '{}'", profile.name);
        self.profile = Some(profile);
    }

    pub fn update_config(&mut self, config: &Config) {
        self.vad.set_threshold(config.vad.threshold);
        self.gate.set_hold_time(config.gate.hold_time_ms);
        self.config.speaker.similarity_threshold = config.speaker.similarity_threshold;
    }
}
```

Changes:
- Removed `use burn::tensor::backend::Backend`
- `Processor<B: Backend>` → `Processor`
- `SileroVad<B>` → `SileroVad`, `EcapaTdnn<B>` → `EcapaTdnn`
- `crate::backend::backend_name()` → `"tract, CPU"` string literal
- All method signatures and logic unchanged

- [ ] **Step 2: Commit**

```bash
git add src/pipeline/processor.rs
git commit -m "refactor: remove Backend generic from Processor"
```

---

## Task 7: Update src/app.rs

**Files:**
- Modify: `src/app.rs`

- [ ] **Step 1: Update app.rs**

Replace the full content of `src/app.rs` with:

```rust
//! Top-level application — eframe App implementation.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::pipeline::processor::{PipelineTelemetry, Processor};
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::enrollment::{EnrollmentSession, EnrollmentState};
use crate::speaker::profile::VoiceProfile;
use crate::ui::ActiveView;
use crate::vad::silero::SileroVad;

#[derive(Clone, Copy)]
enum EnrollmentAction { None, Start, Finalize, Reset }

struct LivePipeline {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    _processor_handle: JoinHandle<()>,
    _stop_signal: Sender<Vec<f32>>,
}

pub struct VoiceGateApp {
    config: Config,
    config_path: PathBuf,
    active_view: ActiveView,
    is_running: bool,
    voice_profile: Option<VoiceProfile>,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    live: Option<LivePipeline>,
    enrollment: Option<EnrollmentSession>,
    last_error: Option<String>,
}

impl VoiceGateApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let config_path = PathBuf::from("config.json");
        let config = Config::load(&config_path);
        let profile_path = config.profiles_dir.join("default.json");
        let voice_profile = VoiceProfile::load(&profile_path).ok();

        if voice_profile.is_some() { log::info!("Loaded voice profile"); }
        else { log::info!("No voice profile — enrollment required"); }

        Self {
            config, config_path,
            active_view: ActiveView::Main,
            is_running: false, voice_profile,
            telemetry: Arc::new(RwLock::new(PipelineTelemetry::default())),
            live: None, enrollment: None, last_error: None,
        }
    }

    fn start(&mut self) {
        if self.is_running { return; }
        match self.try_start() {
            Ok(()) => { self.is_running = true; self.last_error = None; log::info!("Pipeline started"); }
            Err(e) => { self.last_error = Some(format!("Start failed: {}", e)); log::error!("{:#}", e); }
        }
    }

    fn try_start(&mut self) -> anyhow::Result<()> {
        let sr = self.config.audio.sample_rate;
        let fs = self.config.audio.frame_samples;

        // Audio I/O
        let input_dev = crate::audio::capture::default_input_device()?;
        let (audio_tx, audio_rx) = bounded::<Vec<f32>>(64);
        let (input_stream, _) = crate::audio::capture::start_capture(&input_dev, sr, fs, audio_tx.clone())?;

        let output_dev = crate::audio::output::default_output_device()?;
        let (output_tx, output_rx) = bounded::<Vec<f32>>(64);
        let output_stream = crate::audio::output::start_output(&output_dev, sr, output_rx)?;

        // ML Models (tract — loaded from ONNX at runtime)
        let vad = SileroVad::new(
            self.config.vad.threshold,
            Path::new("models/silero_vad.onnx"),
        )?;
        let ecapa = EcapaTdnn::new(
            Path::new("models/ecapa_tdnn.onnx"),
        )?;

        // Processor thread
        let telemetry = self.telemetry.clone();
        let profile = self.voice_profile.clone();
        let config = self.config.clone();

        let handle = std::thread::Builder::new()
            .name("voice-gate-processor".into())
            .spawn(move || {
                let mut proc = Processor::new(config, vad, ecapa, profile, telemetry);
                if let Err(e) = proc.run(audio_rx, output_tx) {
                    log::error!("Processor error: {:#}", e);
                }
            })?;

        self.live = Some(LivePipeline {
            _input_stream: input_stream, _output_stream: output_stream,
            _processor_handle: handle, _stop_signal: audio_tx,
        });
        Ok(())
    }

    fn stop(&mut self) {
        self.live = None;
        self.is_running = false;
        *self.telemetry.write() = PipelineTelemetry::default();
        log::info!("Pipeline stopped");
    }

    fn toggle(&mut self) {
        if self.is_running { self.stop(); } else { self.start(); }
    }
}

impl eframe::App for VoiceGateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running { ctx.request_repaint(); }

        // Nav bar
        egui::TopBottomPanel::top("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_view, ActiveView::Main, "Dashboard");
                ui.selectable_value(&mut self.active_view, ActiveView::Enrollment, "Enrollment");
                ui.selectable_value(&mut self.active_view, ActiveView::Settings, "Settings");
            });
        });

        // Error banner
        let mut clear_error = false;
        if let Some(err) = &self.last_error {
            let err = err.clone();
            egui::TopBottomPanel::top("error").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("Error: {}", err)).color(egui::Color32::from_rgb(220, 60, 60)));
                    if ui.small_button("x").clicked() { clear_error = true; }
                });
            });
        }
        if clear_error { self.last_error = None; }

        // Central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_view {
                ActiveView::Main => {
                    let telem = self.telemetry.clone();
                    let running = self.is_running;
                    let has_profile = self.voice_profile.is_some();
                    crate::ui::main_view::show(ui, &telem, running, has_profile, &mut || self.toggle());
                }
                ActiveView::Enrollment => {
                    if self.enrollment.is_none() {
                        self.enrollment = Some(EnrollmentSession::new(
                            self.config.audio.sample_rate, self.config.speaker.min_enrollment_seconds,
                        ));
                    }
                    let e = self.enrollment.as_ref().unwrap();
                    let state = e.state.clone();
                    let secs = e.speech_seconds();
                    let min = self.config.speaker.min_enrollment_seconds;
                    let action = Cell::new(EnrollmentAction::None);
                    crate::ui::enrollment_view::show(ui, &state, secs, min,
                        &mut || action.set(EnrollmentAction::Start),
                        &mut || action.set(EnrollmentAction::Finalize),
                        &mut || action.set(EnrollmentAction::Reset),
                    );
                    let e = self.enrollment.as_mut().unwrap();
                    match action.get() {
                        EnrollmentAction::None => {}
                        EnrollmentAction::Start => e.start(),
                        EnrollmentAction::Finalize => { e.state = EnrollmentState::Processing; }
                        EnrollmentAction::Reset => e.reset(),
                    }
                }
                ActiveView::Settings => {
                    if crate::ui::settings_view::show(ui, &mut self.config) {
                        let _ = self.config.save(&self.config_path);
                    }
                }
            }
        });
    }
}
```

Changes:
- Removed `use crate::backend::AppBackend`
- `SileroVad::<AppBackend>::new(threshold, &device)` → `SileroVad::new(threshold, Path::new(...))`
- `EcapaTdnn::<AppBackend>::new(&device)` → `EcapaTdnn::new(Path::new(...))`
- Removed `let device = Default::default();`
- Added `use std::path::Path`

- [ ] **Step 2: Commit**

```bash
git add src/app.rs
git commit -m "refactor: update app.rs to use tract model loading"
```

---

## Task 8: Update UI files

**Files:**
- Modify: `src/ui/main_view.rs`
- Modify: `src/ui/settings_view.rs`

- [ ] **Step 1: Update main_view.rs — remove backend_name() line**

In `src/ui/main_view.rs`, replace lines 54-56:

```rust
    // Backend info
    ui.add_space(8.0);
    ui.label(RichText::new(format!("Backend: {}", crate::backend::backend_name())).weak().small());
```

with:

```rust
    ui.add_space(8.0);
    ui.label(RichText::new("Inference: tract (CPU)").weak().small());
```

- [ ] **Step 2: Update settings_view.rs — remove backend/cfg references**

In `src/ui/settings_view.rs`, replace lines 51-62:

```rust
    ui.group(|ui| {
        ui.label(RichText::new("Runtime").strong());
        ui.label(format!("Inference backend: {}", crate::backend::backend_name()));
        #[cfg(has_silero_vad)]
        ui.label("Silero VAD: compiled (Burn native)");
        #[cfg(not(has_silero_vad))]
        ui.label("Silero VAD: NOT compiled (energy fallback)");
        #[cfg(has_ecapa_tdnn)]
        ui.label("ECAPA-TDNN: compiled (Burn native)");
        #[cfg(not(has_ecapa_tdnn))]
        ui.label("ECAPA-TDNN: NOT compiled (speaker verification disabled)");
    });
```

with:

```rust
    ui.group(|ui| {
        ui.label(RichText::new("Runtime").strong());
        ui.label("Inference: tract (CPU, pure Rust)");
        ui.label("Silero VAD: loaded from models/silero_vad.onnx");
        ui.label("ECAPA-TDNN: loaded from models/ecapa_tdnn.onnx");
    });
```

- [ ] **Step 3: Commit**

```bash
git add src/ui/main_view.rs src/ui/settings_view.rs
git commit -m "refactor: update UI to remove Burn backend references"
```

---

## Task 9: Build and verify

- [ ] **Step 1: Run cargo check**

Run: `cargo check 2>&1`

Expected: clean compilation with no errors. Warnings are OK.

If there are errors, fix them before proceeding. Common issues:
- Missing imports (check exact `use` paths)
- Type mismatches in `Output` methods (check tract API for exact method signatures)

- [ ] **Step 2: Run cargo build**

Run: `cargo build 2>&1`

Expected: successful build. The binary won't run without ONNX models, but it should compile.

- [ ] **Step 3: Run cargo clippy (if available)**

Run: `cargo clippy 2>&1 | head -30`

Fix any warnings that relate to the changed files.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve build issues from tract migration"
```

---

## Task 10: Smoke test with models

- [ ] **Step 1: Verify model files exist**

Run: `ls -la models/*.onnx`

If models are missing, download them:
```bash
# Silero VAD
wget -O models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# ECAPA-TDNN (WeSpeaker)
# See README.md for download instructions
```

- [ ] **Step 2: Run the application**

Run: `cargo run 2>&1 | head -20`

Expected: app starts, logs show "Silero VAD loaded" and "ECAPA-TDNN loaded", GUI window opens.

If models fail to load, check tract error messages for unsupported operators.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Burn to tract migration"
```
