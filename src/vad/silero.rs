//! Silero VAD — voice activity detection via tract ONNX runtime.
//!
//! The ONNX model is loaded at runtime from `models/silero_vad.onnx`.
//!
//! Model inputs (3):
//!   - `input`:  `[1, N]`      f32  — audio samples
//!   - `state`:  `[2, 1, 128]` f32  — combined LSTM state (h + c)
//!   - `sr`:     scalar        i64  — sample rate
//!
//! Model outputs (2):
//!   - `output`: `[1, 1]`      f32  — speech probability
//!   - `stateN`: `[2, 1, 128]` f32  — updated LSTM state

use std::path::Path;

use anyhow::Result;

use crate::inference::{DType, Input, InputFact, ModelState, OnnxModel};
use super::VadResult;

/// Number of LSTM layers in the Silero VAD model.
const LSTM_LAYERS: usize = 2;

/// Hidden size per LSTM layer in the Silero VAD model.
const LSTM_HIDDEN_SIZE: usize = 128;

/// Silero VAD wrapper — stateful LSTM model.
pub struct SileroVad {
    threshold: f32,
    model: OnnxModel,
    /// Combined LSTM state [2, 1, 128], carried across frames.
    state: Option<ModelState>,
}

impl SileroVad {
    /// Load Silero VAD from an ONNX file.
    pub fn new(threshold: f32, model_path: &Path) -> Result<Self> {
        let model = OnnxModel::load_with_inputs(model_path, &[
            InputFact { shape: vec![1, 0], dtype: DType::F32 },                            // input: [1, N]
            InputFact { shape: vec![LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE], dtype: DType::F32 }, // state: [2, 1, 128]
            InputFact { shape: vec![], dtype: DType::I64 },                                 // sr: scalar
        ])?;
        log::info!("Silero VAD loaded from {}", model_path.display());

        let mut vad = Self {
            threshold,
            model,
            state: None,
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
        let state = self.state.take()
            .unwrap_or_else(|| ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));

        let mut outputs = self.model.run(vec![
            Input::F32 { shape: vec![1, samples.len()], data: samples.to_vec() },
            Input::State(state),
            Input::I64 { shape: vec![], data: vec![16000] },
        ])?;

        // 2 outputs: [probability, updated_state]
        let new_state = outputs.remove(1).into_state();
        let prob = outputs.remove(0).to_scalar_f32()?;
        self.state = Some(new_state);

        Ok(VadResult {
            speech_probability: prob,
            is_speech: prob >= self.threshold,
        })
    }

    /// Reset LSTM state to zeros.
    pub fn reset(&mut self) {
        self.state = Some(ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}
