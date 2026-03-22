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
