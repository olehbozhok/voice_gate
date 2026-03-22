//! Linear interpolation resampler.

pub fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() { return input.to_vec(); }
    let ratio = from_rate as f64 / to_rate as f64;
    let len = (input.len() as f64 / ratio).ceil() as usize;
    (0..len).map(|i| {
        let pos = i as f64 * ratio;
        let idx = pos as usize;
        let frac = pos - idx as f64;
        if idx + 1 < input.len() {
            (input[idx] as f64 * (1.0 - frac) + input[idx + 1] as f64 * frac) as f32
        } else { input[idx.min(input.len() - 1)] }
    }).collect()
}
