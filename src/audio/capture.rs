//! Microphone capture via cpal.
//!
//! Captures at the device's native sample rate and channels, then converts
//! to 16kHz mono for the ML pipeline.

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::Sender;

use crate::audio::{channels_to_mono, PIPELINE_FRAME_SAMPLES, PIPELINE_SAMPLE_RATE};
use crate::audio::resampler;
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
        .map(|devs| {
            devs.filter_map(|d| d.name().ok())
                .collect()
        })
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
/// Captures at the device's native sample rate and channels, converts to
/// 16kHz mono, and sends 512-sample pipeline frames via `tx`.
pub fn start_capture(device: &Device, tx: Sender<Vec<f32>>) -> Result<Stream> {
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
        native_rate, native_channels, PIPELINE_SAMPLE_RATE,
    );

    // Pre-allocate accumulator for converted (16kHz mono) samples.
    let mut mono_acc: Vec<f32> = Vec::with_capacity(PIPELINE_FRAME_SAMPLES * 4);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            // Step 1: N-channel → mono
            let mono = channels_to_mono(data, native_channels);

            // Step 2: native rate → pipeline rate
            let resampled = resampler::resample(&mono, native_rate, PIPELINE_SAMPLE_RATE);

            // Step 3: accumulate and send pipeline frames
            mono_acc.extend_from_slice(&resampled);
            while mono_acc.len() >= PIPELINE_FRAME_SAMPLES {
                let frame: Vec<f32> = mono_acc.drain(..PIPELINE_FRAME_SAMPLES).collect();
                let _ = tx.try_send(frame);
            }
        },
        |err| log::error!("Input stream error: {}", err),
        None,
    )?;

    stream.play().context("failed to start input stream")?;
    Ok(stream)
}
