//! Silero VAD — voice activity detection via ONNX Runtime.
//!
//! The ONNX model is loaded at runtime from `models/silero_vad.onnx`.
//!
//! The model requires a context window prepended to each audio frame.
//! Each call receives `[1, CONTEXT_SAMPLES + FRAME_SAMPLES]` = `[1, 576]`.
//! The last `CONTEXT_SAMPLES` of each frame are saved for the next call.

use std::path::Path;

use anyhow::Result;

use crate::audio::PIPELINE_SAMPLE_RATE;
use crate::inference::{Input, ModelState, OnnxModel};

/// Expected audio frame size in samples (512 at 16kHz = 32ms).
const FRAME_SAMPLES: usize = 512;

/// Context window size prepended to each frame (64 at 16kHz = 4ms).
/// The model uses this overlapping context for continuity between frames.
const CONTEXT_SAMPLES: usize = 64;

/// Number of LSTM layers in the Silero VAD model.
const LSTM_LAYERS: usize = 2;

/// Hidden size per LSTM layer in the Silero VAD model.
const LSTM_HIDDEN_SIZE: usize = 128;

/// Shape of the combined LSTM state tensor.
const STATE_SHAPE: [usize; 3] = [LSTM_LAYERS, 1, LSTM_HIDDEN_SIZE];

/// Silero VAD wrapper — stateful LSTM model with context window.
///
/// Returns speech probability only. The caller decides the threshold.
pub struct SileroVad {
    model: OnnxModel,
    /// Combined LSTM state [2, 1, 128], carried across frames.
    state: Option<ModelState>,
    /// Context from the end of the previous frame (64 samples).
    context: Vec<f32>,
}

impl SileroVad {
    /// Load Silero VAD from an ONNX file.
    pub fn new(model_path: &Path) -> Result<Self> {
        let model = OnnxModel::load(model_path)?;
        log::info!("Silero VAD loaded from {}", model_path.display());

        let mut vad = Self {
            model,
            state: None,
            context: vec![0.0; CONTEXT_SAMPLES],
        };
        vad.reset();

        // Warmup: run a dummy frame to pre-allocate buffers.
        let dummy = vec![0.0f32; FRAME_SAMPLES];
        let _ = vad.process(&dummy);
        vad.reset();

        Ok(vad)
    }

    /// Run VAD inference on a single audio frame (512 samples at 16 kHz).
    /// Returns only the speech probability. The caller applies the threshold.
    pub fn process(&mut self, samples: &[f32]) -> Result<f32> {
        let state = self.state.take()
            .unwrap_or_else(|| ModelState::zeros_f32(&STATE_SHAPE));

        // Prepend context from previous frame: [context(64) | samples(512)] = 576
        let mut input_with_context = Vec::with_capacity(CONTEXT_SAMPLES + samples.len());
        input_with_context.extend_from_slice(&self.context);
        input_with_context.extend_from_slice(samples);

        // Save the last CONTEXT_SAMPLES of this frame for next call.
        let ctx_start = samples.len().saturating_sub(CONTEXT_SAMPLES);
        self.context = samples[ctx_start..].to_vec();
        if self.context.len() < CONTEXT_SAMPLES {
            let mut padded = vec![0.0; CONTEXT_SAMPLES - self.context.len()];
            padded.extend_from_slice(&self.context);
            self.context = padded;
        }

        let total_len = input_with_context.len();
        let outputs = self.model.run(vec![
            Input::F32 { shape: vec![1, total_len], data: input_with_context },
            Input::State(state),
            Input::I64 { shape: vec![], data: vec![PIPELINE_SAMPLE_RATE as i64] },
        ])?;

        let (prob, new_state) = Self::split_outputs(outputs)?;
        self.state = Some(new_state);

        Ok(prob)
    }

    /// Separate probability output from state output by tensor size.
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

    /// Reset LSTM state and context to zeros.
    pub fn reset(&mut self) {
        self.state = Some(ModelState::zeros_f32(&STATE_SHAPE));
        self.context = vec![0.0; CONTEXT_SAMPLES];
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
        let vad = SileroVad::new(Path::new(MODEL_PATH));
        assert!(vad.is_ok(), "failed to load model: {:?}", vad.err());
    }

    #[test]
    fn silence_has_low_probability() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let mut vad = SileroVad::new(Path::new(MODEL_PATH)).unwrap();

        let silence = vec![0.0f32; FRAME_SAMPLES];
        let prob = vad.process(&silence).unwrap();

        assert!(prob < 0.3, "expected low probability for silence, got {}", prob);
    }

    #[test]
    fn reset_clears_state() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let mut vad = SileroVad::new(Path::new(MODEL_PATH)).unwrap();

        let silence = vec![0.0f32; FRAME_SAMPLES];
        let _ = vad.process(&silence).unwrap();
        let _ = vad.process(&silence).unwrap();

        vad.reset();
        let after_reset = vad.process(&silence).unwrap();

        let mut fresh = SileroVad::new(Path::new(MODEL_PATH)).unwrap();
        let first_frame = fresh.process(&silence).unwrap();

        let diff = (after_reset - first_frame).abs();
        assert!(diff < 1e-5, "reset didn't restore initial state, diff={}", diff);
    }

    /// Verify the model detects speech in recorded audio (requires test_mic.wav).
    #[test]
    fn detects_speech_in_wav() {
        if skip_if_no_model() { eprintln!("SKIP: model not found"); return; }
        let wav_path = Path::new("test_mic.wav");
        if !wav_path.exists() { eprintln!("SKIP: test_mic.wav not found"); return; }

        /// Threshold for speech detection in tests.
        const TEST_THRESHOLD: f32 = 0.5;

        let reader = hound::WavReader::open(wav_path).unwrap();
        let samples: Vec<f32> = reader
            .into_samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();

        let mut vad = SileroVad::new(Path::new(MODEL_PATH)).unwrap();
        let mut speech_frames = 0;
        let mut total_frames = 0;

        for chunk in samples.chunks(FRAME_SAMPLES) {
            if chunk.len() < FRAME_SAMPLES { break; }
            let prob = vad.process(chunk).unwrap();
            if prob >= TEST_THRESHOLD { speech_frames += 1; }
            total_frames += 1;
        }

        eprintln!("Speech frames: {}/{}", speech_frames, total_frames);
        assert!(speech_frames > 0, "VAD detected no speech in test_mic.wav");
    }
}
