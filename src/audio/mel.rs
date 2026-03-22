//! Mel-spectrogram feature extraction for speaker embedding models.
//!
//! Computes log mel-filterbank energies from raw audio, matching
//! the WeSpeaker ECAPA-TDNN model's expected input format:
//! `[batch, num_frames, 80]` mel features.

use rustfft::{num_complex::Complex, FftPlanner};

/// Sample rate expected by the feature extractor.
const SAMPLE_RATE: u32 = 16_000;

/// Frame length in seconds (25ms, standard for speech).
const FRAME_LENGTH_SECS: f32 = 0.025;

/// Hop length in seconds (10ms, standard for speech).
const HOP_LENGTH_SECS: f32 = 0.010;

/// FFT size (next power of 2 above frame length).
const N_FFT: usize = 512;

/// Number of mel filterbank channels.
pub const N_MELS: usize = 80;

/// Minimum frequency for mel filterbank (Hz).
const MEL_FMIN: f32 = 20.0;

/// Small constant to prevent log(0).
const LOG_FLOOR: f32 = 1e-6;

/// Frame length in samples at 16kHz.
const FRAME_SAMPLES: usize = (FRAME_LENGTH_SECS * SAMPLE_RATE as f32) as usize; // 400

/// Hop length in samples at 16kHz.
const HOP_SAMPLES: usize = (HOP_LENGTH_SECS * SAMPLE_RATE as f32) as usize; // 160

/// Compute log mel-filterbank features from raw 16kHz mono audio.
///
/// Returns a flat `Vec<f32>` of shape `[num_frames, N_MELS]` (row-major).
/// The caller should reshape to `[1, num_frames, 80]` for the model.
pub fn compute_mel_features(audio: &[f32]) -> (Vec<f32>, usize) {
    let filterbank = build_mel_filterbank();
    let window = hann_window(FRAME_SAMPLES);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let mut features = Vec::new();
    let mut num_frames = 0;

    let mut start = 0;
    while start + FRAME_SAMPLES <= audio.len() {
        // Apply window
        let mut fft_input: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); N_FFT];
        for (i, &w) in window.iter().enumerate() {
            fft_input[i] = Complex::new(audio[start + i] * w, 0.0);
        }

        // FFT
        fft.process(&mut fft_input);

        // Power spectrum (only first N_FFT/2 + 1 bins)
        let n_bins = N_FFT / 2 + 1;
        let power_spectrum: Vec<f32> = fft_input[..n_bins].iter().map(|c| c.norm_sqr()).collect();

        // Apply mel filterbank + log
        for mel_idx in 0..N_MELS {
            let offset = mel_idx * n_bins;
            let energy: f32 = filterbank[offset..offset + n_bins]
                .iter()
                .zip(power_spectrum.iter())
                .map(|(&f, &p)| f * p)
                .sum();
            features.push((energy + LOG_FLOOR).ln());
        }

        num_frames += 1;
        start += HOP_SAMPLES;
    }

    (features, num_frames)
}

/// Convert frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Build triangular mel filterbank matrix (flat, row-major `[N_MELS, N_FFT/2+1]`).
fn build_mel_filterbank() -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1;
    let mel_max = hz_to_mel(SAMPLE_RATE as f32 / 2.0);
    let mel_min = hz_to_mel(MEL_FMIN);

    // N_MELS + 2 equally spaced points on the mel scale
    let mel_points: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((N_FFT as f32 + 1.0) * hz / SAMPLE_RATE as f32).floor() as usize)
        .collect();

    let mut filterbank = vec![0.0f32; N_MELS * n_bins];

    for i in 0..N_MELS {
        let offset = i * n_bins;
        // Rising slope
        if bin_points[i + 1] > bin_points[i] {
            for j in bin_points[i]..bin_points[i + 1] {
                if j < n_bins {
                    filterbank[offset + j] =
                        (j - bin_points[i]) as f32 / (bin_points[i + 1] - bin_points[i]) as f32;
                }
            }
        }
        // Falling slope
        if bin_points[i + 2] > bin_points[i + 1] {
            for j in bin_points[i + 1]..bin_points[i + 2] {
                if j < n_bins {
                    filterbank[offset + j] = (bin_points[i + 2] - j) as f32
                        / (bin_points[i + 2] - bin_points[i + 1]) as f32;
                }
            }
        }
    }

    filterbank
}

/// Generate a Hann window of given length.
fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / len as f32).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_features_shape() {
        // 1 second of silence at 16kHz
        let audio = vec![0.0f32; 16000];
        let (features, num_frames) = compute_mel_features(&audio);

        // With 400-sample frames and 160-sample hops:
        // (16000 - 400) / 160 + 1 = 98 frames
        assert!(
            num_frames > 90 && num_frames < 110,
            "expected ~98 frames for 1s audio, got {}",
            num_frames
        );
        assert_eq!(features.len(), num_frames * N_MELS);
    }

    #[test]
    fn mel_features_non_zero_for_signal() {
        // Generate a tone
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let (features, _) = compute_mel_features(&audio);

        let max_val = features.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = features.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_val > min_val, "features should vary for a tone signal");
    }
}
