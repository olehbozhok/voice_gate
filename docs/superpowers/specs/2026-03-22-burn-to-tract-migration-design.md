# Burn to Tract Migration Design

**Date:** 2026-03-22
**Status:** Draft

## Summary

Replace the Burn deep learning framework with tract for ONNX model inference. This removes build-time ONNX-to-Rust code generation, eliminates the `B: Backend` generic parameter, and switches to runtime model loading via `tract-onnx`. Pure Rust, CPU-only, no external runtime dependencies.

## Motivation

- `burn-import` does not support all ONNX operators used by Silero VAD and ECAPA-TDNN, causing build failures
- Build-time code generation adds complexity (build.rs, generated model includes, cfg flags)
- The `B: Backend` generic parameter threads through the entire codebase unnecessarily for CPU-only inference
- tract has broad ONNX operator coverage and loads models at runtime with no build step

## Approach

Runtime model loading (Approach A). ONNX files are loaded and optimized at application startup. No build-time processing.

## Dependency Changes

### Remove

| Crate | Role |
|-------|------|
| `burn` | ML framework (with ndarray/wgpu/cuda features) |
| `burn-import` | Build-time ONNX-to-Rust code generation |

### Add

| Crate | Version | Role |
|-------|---------|------|
| `tract-onnx` | `0.22` | ONNX model loading and inference |

### Feature Flag Changes

| Flag | Before | After |
|------|--------|-------|
| `cpu` | Select NdArray backend | **Remove** |
| `wgpu` | Select Wgpu backend | **Remove** |
| `cuda` | Select Cuda backend | **Remove** |
| `dev` | Lenient build (fallbacks) | **Remove** |

## Architecture Changes

### Before (Burn)

```
build.rs (burn-import)
  compile ONNX -> Rust code at build time
  set cfg flags: has_silero_vad, has_ecapa_tdnn

src/model/mod.rs
  include!(generated code)

src/backend.rs
  type AppBackend = NdArray | Wgpu | Cuda  (feature-selected)

SileroVad<B: Backend>  ->  Tensor<B, 2>  ->  model.forward()
EcapaTdnn<B: Backend>  ->  Tensor<B, 2>  ->  model.forward()
Processor<B: Backend>
```

### After (tract)

```
models/silero_vad.onnx    (loaded at runtime)
models/ecapa_tdnn.onnx    (loaded at runtime)

src/inference.rs           OnnxModel wrapper (hides tract)
SileroVad  ->  OnnxModel  ->  tract::Tensor  ->  model.run()
EcapaTdnn  ->  OnnxModel  ->  tract::Tensor  ->  model.run()
Processor  (no generic parameter)
```

## Component Designs

### OnnxModel (inference wrapper)

New file: `src/inference.rs` — thin wrapper over the ONNX runtime. No tract types in the public API. To swap runtimes later, change only this file.

```rust
use std::path::Path;

/// Wrapper over an ONNX inference runtime.
/// Currently backed by tract. To swap runtimes, change only this file.
pub struct OnnxModel { /* tract internals hidden */ }

/// Opaque handle for model state (e.g. LSTM h/c).
/// Zero-copy round-trip between inference calls.
pub struct ModelState { /* wraps tract Tensor */ }

/// Typed model input. No framework-specific types exposed.
pub enum Input {
    F32 { shape: Vec<usize>, data: Vec<f32> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
    State(ModelState),
}

/// Opaque model output with typed accessors.
pub struct Output { /* wraps tract TValue */ }

impl Output {
    pub fn to_scalar_f32(&self) -> Result<f32>
    pub fn to_f32_vec(&self) -> Result<Vec<f32>>
    pub fn into_state(self) -> ModelState   // zero-copy for LSTM carry
}

impl ModelState {
    pub fn zeros_f32(shape: &[usize]) -> Self
}

impl OnnxModel {
    /// Load an ONNX model from disk, optimize, and prepare for inference.
    pub fn load(path: &Path) -> Result<Self>

    /// Run inference with typed inputs, get typed outputs.
    pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Output>>
}
```

Consumers (`SileroVad`, `EcapaTdnn`) use only types from `inference.rs` — zero tract imports outside this module.

### SileroVad

```rust
use crate::inference::{OnnxModel, ModelState, Input};

pub struct SileroVad {
    threshold: f32,
    model: OnnxModel,
    h: ModelState,  // [2, 1, 64] LSTM hidden state
    c: ModelState,  // [2, 1, 64] LSTM cell state
}

impl SileroVad {
    /// Load Silero VAD from ONNX file via OnnxModel.
    pub fn new(threshold: f32, model_path: &Path) -> Result<Self>

    /// Run one frame through VAD. Stateful (updates h, c).
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult>

    /// Reset LSTM state to zeros.
    pub fn reset(&mut self)
}
```

**Input/output per frame:**
- Input: `[1, num_samples]` f32 (typically 512 samples = 32ms at 16kHz)
- Sample rate: `[1]` i64 = 16000
- Hidden state h: `ModelState` [2, 1, 64] f32
- Cell state c: `ModelState` [2, 1, 64] f32
- Output: speech probability scalar f32 + updated h, c as `ModelState`

**Inference:**
```rust
let outputs = self.model.run(vec![
    Input::F32 { shape: vec![1, samples.len()], data: samples.to_vec() },
    Input::I64 { shape: vec![1], data: vec![16000] },
    Input::State(std::mem::take(&mut self.h)),
    Input::State(std::mem::take(&mut self.c)),
])?;
let prob = outputs[0].to_scalar_f32()?;
self.h = outputs[1].into_state();
self.c = outputs[2].into_state();
```

### EcapaTdnn

```rust
use crate::inference::{OnnxModel, Input};

pub struct EcapaTdnn {
    model: OnnxModel,
}

impl EcapaTdnn {
    /// Load ECAPA-TDNN from ONNX file via OnnxModel.
    pub fn new(model_path: &Path) -> Result<Self>

    /// Extract 192-dim L2-normalized speaker embedding from waveform.
    pub fn extract(&self, waveform: &[f32]) -> Result<Vec<f32>>
}
```

**Input/output:**
- Input: `[1, num_samples]` f32 (typically 24,000 samples = 1.5s at 16kHz)
- Output: `Vec<f32>` [192] L2-normalized via `output.to_f32_vec()`

### Processor

Remove the `B: Backend` generic parameter:

```rust
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
```

All method signatures stay the same. Only internal types change.

### App startup

```rust
fn try_start(&mut self) -> anyhow::Result<()> {
    // Model loading (was: SileroVad::<AppBackend>::new(threshold, &device))
    let vad = SileroVad::new(
        self.config.vad.threshold,
        Path::new("models/silero_vad.onnx"),
    )?;
    let ecapa = EcapaTdnn::new(
        Path::new("models/ecapa_tdnn.onnx"),
    )?;

    // ... rest unchanged (audio setup, processor thread spawn)
}
```

Model paths come from the `models/` directory relative to the current working directory. If a model file is missing, `SileroVad::new()` / `EcapaTdnn::new()` return an error with a clear message, which propagates to the UI error banner.

**Warmup:** After loading each model, run a single dummy inference to pre-allocate internal buffers. This avoids a latency spike on the first real audio frame.

## Files to Delete

| File | Reason |
|------|--------|
| `build.rs` | No build-time ONNX compilation |
| `src/backend.rs` | No `AppBackend` type alias |
| `src/model/mod.rs` | No generated model includes |

## Files to Create

| File | Purpose |
|------|---------|
| `src/inference.rs` | `OnnxModel` wrapper — thin abstraction over tract runtime |

## Files to Modify

| File | Changes |
|------|---------|
| `Cargo.toml` | Remove burn/burn-import, add tract-onnx, remove feature flags |
| `src/main.rs` | Remove `mod model`, `mod backend`, add `mod inference`, remove `backend_name()` log call |
| `src/vad/silero.rs` | Rewrite: tract tensors, runtime model loading |
| `src/speaker/embedding.rs` | Rewrite: tract tensors, runtime model loading |
| `src/pipeline/processor.rs` | Remove `B: Backend` generic, remove `backend_name()` log call |
| `src/app.rs` | Pass model paths, remove `AppBackend` references |
| `src/ui/main_view.rs` | Remove `backend_name()` display call |
| `src/ui/settings_view.rs` | Remove `backend_name()` call, remove `has_silero_vad`/`has_ecapa_tdnn` cfg conditionals, update info labels to reflect tract |

| `src/error.rs` | Add model loading error variants |

## Files Unchanged

| File | Reason |
|------|--------|
| `src/audio/*` | No Burn dependency |
| `src/pipeline/state_machine.rs` | Pure logic |
| `src/speaker/profile.rs` | Pure data |
| `src/speaker/enrollment.rs` | Orchestration only |
| `src/ui/*` (except `main_view.rs`, `settings_view.rs`) | UI layer |
| `src/config.rs` | No Burn dependency |

## Dev Fallbacks

The `dev` feature and all fallbacks (energy-based VAD, zero embeddings) are removed. Models are always required at runtime. If models are missing, the app starts but shows a clear error when the user tries to start the pipeline. This is simpler and avoids shipping untested fallback code paths.

## Error Handling

Model loading errors surface as `anyhow::Error` at startup. The UI error banner already displays startup failures. No new error UI needed.

New error cases:
- ONNX file not found → "Model not found: models/silero_vad.onnx"
- ONNX parse failure → "Failed to load model: {tract error}"
- Inference failure → "Inference error: {tract error}"

## Testing Strategy

1. Build succeeds without models (compilation only, no runtime)
2. App starts and shows clear error if models are missing
3. With valid ONNX models: VAD produces speech probabilities, ECAPA-TDNN produces 192-dim embeddings
4. End-to-end: gate opens for enrolled speaker, stays closed for others
