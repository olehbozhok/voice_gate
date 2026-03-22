//! Audio I/O layer via cpal.
pub mod capture;
pub mod mel;
pub mod output;
pub mod resampler;

/// Sample rate expected by the ML pipeline (Silero VAD, ECAPA-TDNN).
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Number of mono samples per pipeline frame (32ms at 16kHz).
pub const PIPELINE_FRAME_SAMPLES: usize = 512;

/// Pipeline channel count (mono).
pub const PIPELINE_CHANNELS: u16 = 1;

/// Duration of the output ring buffer in seconds. Caps the deque to prevent
/// unbounded growth if the consumer (cpal callback) falls behind.
pub const OUTPUT_RING_BUFFER_SECS: f32 = 1.0;

/// RMS energy of a frame.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Convert interleaved N-channel audio to mono by averaging all channels.
pub fn channels_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Expand mono audio to N interleaved channels by duplicating each sample.
pub fn mono_to_channels(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    let mut out = Vec::with_capacity(samples.len() * ch);
    for &s in samples {
        for _ in 0..ch {
            out.push(s);
        }
    }
    out
}
