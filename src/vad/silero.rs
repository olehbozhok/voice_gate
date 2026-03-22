//! Silero VAD — voice activity detection via ONNX Runtime.
//!
//! The ONNX model is loaded at runtime from `models/silero_vad.onnx`.
//!
//! Model inputs (3):
//!   - `input`:  `[1, N]`      f32  — audio samples
//!   - `state`:  `[2, 1, 128]` f32  — combined LSTM state
//!   - `sr`:     scalar        i64  — sample rate
//!
//! Model outputs (2):
//!   - `output`: `[1, 1]`      f32  — speech probability
//!   - `stateN`: `[2, 1, 128]` f32  — updated LSTM state

use std::path::Path;

use anyhow::Result;

use crate::audio::PIPELINE_SAMPLE_RATE;
use crate::inference::{Input, ModelState, OnnxModel};
use super::VadResult;

/// Expected audio frame size in samples (512 at 16kHz = 32ms).
const FRAME_SAMPLES: usize = 512;

/// Number of LSTM layers in the Silero VAD model.
const LSTM_LAYERS: usize = 2;

/// Hidden size per LSTM layer in the Silero VAD model.
const LSTM_HIDDEN_SIZE: usize = 128;

/// Shape of the combined LSTM state tensor.
const STATE_SHAPE: [usize; 3] = [LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE];

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
        let model = OnnxModel::load(model_path)?;
        log::info!("Silero VAD loaded from {}", model_path.display());

        let mut vad = Self {
            threshold,
            model,
            state: None,
        };
        vad.reset();

        // Warmup: run a dummy frame to pre-allocate buffers.
        let dummy = vec![0.0f32; FRAME_SAMPLES];
        let _ = vad.process(&dummy);
        vad.reset();

        Ok(vad)
    }

    /// Run VAD inference on a single audio frame (512 samples at 16 kHz).
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult> {
        let state = self.state.take()
            .unwrap_or_else(|| ModelState::zeros_f32(&STATE_SHAPE));

        let outputs = self.model.run(vec![
            Input::F32 { shape: vec![1, samples.len()], data: samples.to_vec() },
            Input::State(state),
            Input::I64 { shape: vec![], data: vec![PIPELINE_SAMPLE_RATE as i64] },
        ])?;

        // Silero VAD outputs: "output" (probability) and "stateN" (LSTM state).
        // Find the probability output (small tensor) vs state (large tensor).
        let (prob, new_state) = Self::split_outputs(outputs)?;
        self.state = Some(new_state);

        Ok(VadResult {
            speech_probability: prob,
            is_speech: prob >= self.threshold,
        })
    }

    /// Separate probability output from state output.
    /// The probability tensor is small (1-2 elements), the state tensor is large (256 elements).
    fn split_outputs(outputs: Vec<crate::inference::Output>) -> Result<(f32, ModelState)> {
        /// Number of elements in the LSTM state tensor (2 * 1 * 128 = 256).
        const STATE_ELEMENTS: usize = LSTM_LAYERS * 1 * LSTM_HIDDEN_SIZE;

        let mut prob = None;
        let mut state = None;

        for output in outputs {
            let data = output.to_f32_vec()?;
            if data.len() == STATE_ELEMENTS && state.is_none() {
                state = Some(ModelState::from_data(data, STATE_SHAPE.to_vec()));
            } else if prob.is_none() {
                prob = Some(data.first().copied().unwrap_or(0.0));
            }
        }

        let p = prob.ok_or_else(|| anyhow::anyhow!("no probability output found"))?;
        let s = state.ok_or_else(|| anyhow::anyhow!("no state output found"))?;

        log::trace!("VAD prob={:.4}", p);
        Ok((p, s))
    }

    /// Reset LSTM state to zeros.
    pub fn reset(&mut self) {
        self.state = Some(ModelState::zeros_f32(&STATE_SHAPE));
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const MODEL_PATH: &str = "models/silero_vad.onnx";

    fn skip_if_no_model() -> bool {
        !Path::new(MODEL_PATH).exists()
    }

    #[test]
    fn load_model() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let vad = SileroVad::new(0.5, Path::new(MODEL_PATH));
        assert!(vad.is_ok(), "failed to load model: {:?}", vad.err());
    }

    #[test]
    fn silence_has_low_probability() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let mut vad = SileroVad::new(0.5, Path::new(MODEL_PATH)).unwrap();

        let silence = vec![0.0f32; FRAME_SAMPLES];
        let result = vad.process(&silence).unwrap();

        assert!(result.speech_probability < 0.3,
            "expected low probability for silence, got {}", result.speech_probability);
        assert!(!result.is_speech);
    }

    #[test]
    fn state_carries_across_frames() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let mut vad = SileroVad::new(0.5, Path::new(MODEL_PATH)).unwrap();

        let silence = vec![0.0f32; FRAME_SAMPLES];
        let r1 = vad.process(&silence).unwrap();
        let r2 = vad.process(&silence).unwrap();

        assert!(r1.speech_probability < 0.5);
        assert!(r2.speech_probability < 0.5);
    }

    #[test]
    fn reset_clears_state() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let mut vad = SileroVad::new(0.5, Path::new(MODEL_PATH)).unwrap();

        let silence = vec![0.0f32; FRAME_SAMPLES];
        let _ = vad.process(&silence).unwrap();
        let _ = vad.process(&silence).unwrap();

        vad.reset();
        let after_reset = vad.process(&silence).unwrap();

        let mut fresh = SileroVad::new(0.5, Path::new(MODEL_PATH)).unwrap();
        let first_frame = fresh.process(&silence).unwrap();

        let diff = (after_reset.speech_probability - first_frame.speech_probability).abs();
        assert!(diff < 1e-5, "reset didn't restore initial state, diff={}", diff);
    }

    #[test]
    fn threshold_controls_is_speech() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }

        let mut vad = SileroVad::new(0.0, Path::new(MODEL_PATH)).unwrap();
        let silence = vec![0.0f32; FRAME_SAMPLES];
        let result = vad.process(&silence).unwrap();
        assert!(result.is_speech, "threshold=0.0 should classify everything as speech");

        let mut vad = SileroVad::new(1.0, Path::new(MODEL_PATH)).unwrap();
        let result = vad.process(&silence).unwrap();
        assert!(!result.is_speech, "threshold=1.0 should classify nothing as speech");
    }
}
