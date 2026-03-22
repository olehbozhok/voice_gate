//! ECAPA-TDNN speaker embedding extractor — Burn native.
//!
//! # Compile-time behaviour
//!
//! - **Strict mode (default):** missing `ecapa_tdnn.onnx` or conversion failure
//!   stops the build with actionable instructions via `diagnostics.rs`.
//! - **Dev mode (`--features dev`):** returns zero embeddings (speaker
//!   verification effectively disabled — all speech passes through).

use anyhow::Result;
use burn::tensor::{backend::Backend, Tensor};

// Safety net: strict mode without compiled model should have been caught
// by diagnostics.rs already. This catches edge cases.
#[cfg(all(not(has_ecapa_tdnn), not(feature = "dev")))]
compile_error!(
    "\n\nECAPA-TDNN model not compiled but `dev` feature is not enabled.\n\
     This should have been caught by build.rs. Something went wrong.\n\
     Either place models/ecapa_tdnn.onnx and rebuild, or use --features dev.\n\n"
);

/// Wrapper around the ECAPA-TDNN Burn model.
pub struct EcapaTdnn<B: Backend> {
    /// The compiled Burn model.
    #[cfg(has_ecapa_tdnn)]
    model: crate::model::ecapa_tdnn::Model<B>,
    /// Device for inference.
    device: B::Device,
}

impl<B: Backend> EcapaTdnn<B> {
    /// Load the model. Weights are embedded from build-time compilation.
    pub fn new(device: &B::Device) -> Result<Self> {
        #[cfg(has_ecapa_tdnn)]
        let model = {
            let m: crate::model::ecapa_tdnn::Model<B> =
                crate::model::ecapa_tdnn::Model::default();
            log::info!("ECAPA-TDNN loaded (Burn-compiled, native Rust)");
            m
        };

        #[cfg(not(has_ecapa_tdnn))]
        log::warn!("ECAPA-TDNN: not compiled (dev mode) — speaker verification disabled");

        Ok(Self {
            #[cfg(has_ecapa_tdnn)]
            model,
            device: device.clone(),
        })
    }

    /// Extract a speaker embedding from an audio waveform.
    ///
    /// # Arguments
    ///
    /// * `waveform` — Mono f32 audio at 16 kHz. Ideally 2–5 seconds.
    ///
    /// # Returns
    ///
    /// An L2-normalized embedding vector.
    pub fn extract(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        #[cfg(has_ecapa_tdnn)]
        {
            self.extract_neural(waveform)
        }

        #[cfg(not(has_ecapa_tdnn))]
        {
            // Dev mode: return zero embedding → cosine_similarity ≈ 0 → no match.
            // This means all speech passes through (no speaker gating).
            log::trace!("ECAPA-TDNN not compiled (dev mode) — returning zero embedding");
            Ok(vec![0.0; 192])
        }
    }

    /// Neural embedding extraction via the compiled Burn model.
    #[cfg(has_ecapa_tdnn)]
    fn extract_neural(&self, waveform: &[f32]) -> Result<Vec<f32>> {
        let num_samples = waveform.len();

        if num_samples < 16000 {
            log::warn!(
                "Waveform too short ({} samples = {:.1}s), embedding may be unreliable",
                num_samples,
                num_samples as f32 / 16000.0
            );
        }

        // Create input tensor: [1, num_samples].
        let input = Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::from(waveform).convert::<f32>(),
            &self.device,
        )
        .reshape([1, num_samples]);

        // Forward pass.
        let output = self.model.forward(input);

        // Extract embedding as Vec<f32>.
        let data = output.to_data();
        let embedding: Vec<f32> = data
            .as_slice::<f32>()
            .unwrap_or(&[])
            .to_vec();

        // L2-normalize.
        Ok(l2_normalize(&embedding))
    }

    /// Returns `true` if the neural model is available.
    pub fn is_available(&self) -> bool {
        cfg!(has_ecapa_tdnn)
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
