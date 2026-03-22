//! Silero VAD — voice activity detection via Burn.
//!
//! The ONNX model is converted to native Burn code at build time by `burn-onnx`.
//! At runtime it's a plain Rust struct — no ONNX Runtime, no C++ dependencies.
//!
//! # Compile-time behaviour
//!
//! - **Strict mode (default):** if `silero_vad.onnx` is missing or fails to
//!   convert, `build.rs` emits a `compile_error!` with download instructions.
//! - **Dev mode (`--features dev`):** falls back to energy-based detection.

use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor};

use super::VadResult;

// If we're not in dev mode and the model isn't compiled, the build should
// have already stopped via diagnostics.rs. This is a safety net.
#[cfg(all(not(has_silero_vad), not(feature = "dev")))]
compile_error!(
    "\n\nSilero VAD model not compiled but `dev` feature is not enabled.\n\
     This should have been caught by build.rs. Something went wrong.\n\
     Either place models/silero_vad.onnx and rebuild, or use --features dev.\n\n"
);

/// Silero VAD wrapper — neural network or energy-based fallback.
pub struct SileroVad<B: Backend> {
    threshold: f32,
    /// LSTM hidden state (carried across frames).
    h: Tensor<B, 3>,
    /// LSTM cell state (carried across frames).
    c: Tensor<B, 3>,
    #[cfg(has_silero_vad)]
    model: crate::model::silero_vad::Model<B>,
    device: B::Device,
}

impl<B: Backend> SileroVad<B> {
    pub fn new(threshold: f32, device: &B::Device) -> Result<Self> {
        #[cfg(has_silero_vad)]
        let model = {
            let m: crate::model::silero_vad::Model<B> =
                crate::model::silero_vad::Model::default();
            log::info!("Silero VAD loaded (Burn-compiled, native Rust)");
            m
        };

        #[cfg(not(has_silero_vad))]
        log::warn!("Silero VAD: using energy-based fallback (dev mode)");

        Ok(Self {
            threshold,
            h: Tensor::zeros([2, 1, 64], device),
            c: Tensor::zeros([2, 1, 64], device),
            #[cfg(has_silero_vad)]
            model,
            device: device.clone(),
        })
    }

    /// Run VAD inference on a single audio frame (512 samples at 16kHz).
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult> {
        #[cfg(has_silero_vad)]
        { self.process_neural(samples) }

        #[cfg(not(has_silero_vad))]
        { Ok(self.process_energy(samples)) }
    }

    #[cfg(has_silero_vad)]
    fn process_neural(&mut self, samples: &[f32]) -> Result<VadResult> {
        let num_samples = samples.len();

        let input = Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::from(samples).convert::<f32>(),
            &self.device,
        ).reshape([1, num_samples]);

        let sr = Tensor::<B, 1>::from_floats(
            burn::tensor::TensorData::from(&[16000.0f32][..]).convert::<f32>(),
            &self.device,
        );

        let output = self.model.forward(input, sr, self.h.clone(), self.c.clone());

        let prob_data = output.output.to_data();
        let prob: f32 = prob_data.as_slice::<f32>().unwrap_or(&[0.0])[0];

        self.h = output.hn;
        self.c = output.cn;

        Ok(VadResult { speech_probability: prob, is_speech: prob >= self.threshold })
    }

    /// Energy-based fallback (only available with `--features dev`).
    #[cfg(not(has_silero_vad))]
    fn process_energy(&self, samples: &[f32]) -> VadResult {
        let rms = crate::audio::rms(samples);
        let pseudo_prob = 1.0 / (1.0 + (-50.0 * (rms - 0.015)).exp());
        VadResult { speech_probability: pseudo_prob, is_speech: pseudo_prob >= self.threshold }
    }

    pub fn reset(&mut self) {
        self.h = Tensor::zeros([2, 1, 64], &self.device);
        self.c = Tensor::zeros([2, 1, 64], &self.device);
    }

    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }
}
