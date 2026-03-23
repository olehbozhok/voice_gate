//! Audio output via cpal.
//!
//! Receives native-format audio from the processor (already at device rate
//! and channel count) and plays it directly — no resampling needed.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::Receiver;
use parking_lot::Mutex;

use crate::audio::OUTPUT_RING_BUFFER_SECS;
use crate::error::AudioError;

/// Return the system default output device.
pub fn default_output_device() -> Result<Device> {
    cpal::default_host()
        .default_output_device()
        .ok_or_else(|| AudioError::NoOutputDevice.into())
}

/// List the names of all available output devices.
pub fn list_output_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.output_devices()
        .map(|devs| {
            devs.filter_map(|d| d.description().ok().map(|desc| desc.name().to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Find an output device by name. Falls back to default if not found.
pub fn find_output_device(name: &str) -> Result<Device> {
    let host = cpal::default_host();
    if let Ok(devices) = host.output_devices() {
        for dev in devices {
            if dev
                .description()
                .ok()
                .map(|d| d.name().to_string())
                .as_deref()
                == Some(name)
            {
                return Ok(dev);
            }
        }
    }
    log::warn!("Output device '{}' not found, using default", name);
    default_output_device()
}

/// Start playing audio on `device`.
///
/// Receives native-format interleaved samples via `rx` and plays them
/// directly through the device. No resampling or channel expansion needed —
/// the processor already provides audio in the correct format.
pub fn start_output(device: &Device, rx: Receiver<Vec<f32>>) -> Result<Stream> {
    let supported = device
        .default_output_config()
        .context("failed to query output device config")?;

    let native_rate = supported.sample_rate();
    let native_channels = supported.channels();

    let config = StreamConfig {
        channels: native_channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    log::info!(
        "Output: device native config = {}Hz, {} ch",
        native_rate,
        native_channels,
    );

    // Ring buffer cap: 1 second of native-format audio.
    let ring_buffer_cap =
        (native_rate as f32 * native_channels as f32 * OUTPUT_RING_BUFFER_SECS) as usize;

    let buf = Arc::new(Mutex::new(VecDeque::<f32>::with_capacity(ring_buffer_cap)));

    // Feeder thread: push native audio directly to the ring buffer.
    let bp = buf.clone();
    std::thread::Builder::new()
        .name("audio-feeder".into())
        .spawn(move || {
            while let Ok(frame) = rx.recv() {
                let mut b = bp.lock();
                b.extend(frame);
                // Cap the ring buffer to prevent unbounded growth.
                while b.len() > ring_buffer_cap {
                    b.pop_front();
                }
            }
        })?;

    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            let mut b = buf.lock();
            for s in data.iter_mut() {
                *s = b.pop_front().unwrap_or(0.0);
            }
        },
        |err| log::error!("Output error: {}", err),
        None,
    )?;

    stream.play().context("failed to start output")?;
    Ok(stream)
}
