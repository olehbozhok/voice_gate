//! Microphone capture via cpal.
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Stream, StreamConfig};
use crossbeam_channel::Sender;
use crate::error::AudioError;

pub fn default_input_device() -> Result<Device> {
    let host: Host = cpal::default_host();
    host.default_input_device().ok_or_else(|| AudioError::NoInputDevice.into())
}

pub fn start_capture(
    device: &Device, sample_rate: u32, frame_samples: usize, tx: Sender<Vec<f32>>,
) -> Result<(Stream, StreamConfig)> {
    let config = StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let mut acc: Vec<f32> = Vec::with_capacity(frame_samples * 2);
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            acc.extend_from_slice(data);
            while acc.len() >= frame_samples {
                let frame: Vec<f32> = acc.drain(..frame_samples).collect();
                let _ = tx.try_send(frame);
            }
        },
        |err| log::error!("Input stream error: {}", err),
        None,
    )?;
    stream.play().context("failed to start input stream")?;
    log::info!("Capture started ({}Hz, {} samples/frame)", sample_rate, frame_samples);
    Ok((stream, config))
}
