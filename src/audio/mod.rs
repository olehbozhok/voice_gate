//! Audio I/O layer via cpal.
pub mod capture;
pub mod output;
pub mod resampler;

pub type AudioFrame = Vec<f32>;

/// RMS energy of a frame.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}
