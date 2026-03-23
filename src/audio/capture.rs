//! Microphone capture via cpal.
//!
//! Captures at the device's native sample rate and channels, then sends
//! both the original audio and a 16kHz mono downsampled version for ML.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::Sender;

use crate::audio::resampler;
use crate::audio::{channels_to_mono, AudioFrame, PIPELINE_FRAME_SAMPLES, PIPELINE_SAMPLE_RATE};
use crate::error::AudioError;

/// Return the system default input device.
pub fn default_input_device() -> Result<Device> {
    cpal::default_host()
        .default_input_device()
        .ok_or_else(|| AudioError::NoInputDevice.into())
}

/// List the names of all available input devices.
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devs| devs.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

/// Find an input device by name. Falls back to default if not found.
pub fn find_input_device(name: &str) -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(devices) = host.input_devices() {
        for dev in devices {
            if dev.name().ok().as_deref() == Some(name) {
                return Ok(dev);
            }
        }
    }
    log::warn!("Input device '{}' not found, using default", name);
    default_input_device()
}

/// Start capturing audio from `device`.
///
/// Captures at the device's native sample rate and channels. Each callback
/// produces [`AudioFrame`]s containing both the original interleaved audio
/// and 16kHz mono downsampled frames for the ML pipeline.
pub fn start_capture(device: &Device, tx: Sender<AudioFrame>) -> Result<Stream> {
    let supported = device
        .default_input_config()
        .context("failed to query input device config")?;

    let native_rate = supported.sample_rate().0;
    let native_channels = supported.channels();

    let config = StreamConfig {
        channels: native_channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    log::info!(
        "Capture: device native config = {}Hz, {} ch (pipeline: {}Hz mono)",
        native_rate,
        native_channels,
        PIPELINE_SAMPLE_RATE,
    );

    // Ratio: how many native interleaved samples correspond to one pipeline sample.
    // e.g. 48kHz stereo → 16kHz mono: ratio = (48000 * 2) / 16000 = 6.0
    let native_samples_per_pipeline_sample =
        (native_rate as f64 * native_channels as f64) / PIPELINE_SAMPLE_RATE as f64;
    // Number of original interleaved samples that correspond to one pipeline frame.
    let original_frame_size =
        (PIPELINE_FRAME_SAMPLES as f64 * native_samples_per_pipeline_sample).round() as usize;

    // Accumulators for building aligned frames.
    let mut mono_acc: Vec<f32> = Vec::with_capacity(PIPELINE_FRAME_SAMPLES * 4);
    let mut original_acc: Vec<f32> = Vec::with_capacity(original_frame_size * 4);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Accumulate original interleaved samples.
            original_acc.extend_from_slice(data);

            // Downsample for pipeline: N-channel → mono → 16kHz.
            let mono = channels_to_mono(data, native_channels);
            let resampled = resampler::resample(&mono, native_rate, PIPELINE_SAMPLE_RATE);
            mono_acc.extend_from_slice(&resampled);

            // Emit aligned frames: one pipeline frame + corresponding original chunk.
            while mono_acc.len() >= PIPELINE_FRAME_SAMPLES
                && original_acc.len() >= original_frame_size
            {
                let pipeline: Vec<f32> = mono_acc.drain(..PIPELINE_FRAME_SAMPLES).collect();
                let original: Vec<f32> = original_acc.drain(..original_frame_size).collect();
                let _ = tx.try_send(AudioFrame { pipeline, original });
            }
        },
        |err| log::error!("Input stream error: {}", err),
        None,
    )?;

    stream.play().context("failed to start input stream")?;
    Ok(stream)
}
