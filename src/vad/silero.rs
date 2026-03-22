//! Silero VAD — voice activity detection via tract ONNX runtime.
//!
//! The ONNX model is loaded at runtime from `models/silero_vad.onnx`.

use std::path::Path;

use anyhow::Result;

use crate::inference::{DType, Input, InputFact, ModelState, OnnxModel};
use super::VadResult;

/// Number of LSTM layers in the Silero VAD model.
const LSTM_LAYERS: usize = 2;

/// Hidden size per LSTM layer in the Silero VAD model.
const LSTM_HIDDEN_SIZE: usize = 64;

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
        let model = OnnxModel::load_with_inputs(model_path, &[
            InputFact { shape: vec![1, 0], dtype: DType::F32 },             // input: [1, N] audio
            InputFact { shape: vec![1], dtype: DType::I64 },                // sr: [1] sample rate
            InputFact { shape: vec![LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE], dtype: DType::F32 }, // h
            InputFact { shape: vec![LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE], dtype: DType::F32 }, // c
        ])?;
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
        let h = self.h.take().unwrap_or_else(|| ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));
        let c = self.c.take().unwrap_or_else(|| ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));

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
        self.h = Some(ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));
        self.c = Some(ModelState::zeros_f32(&[LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE]));
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}
