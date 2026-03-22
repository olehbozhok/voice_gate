//! Audio output via cpal.
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use crossbeam_channel::Receiver;
use crate::error::AudioError;

pub fn default_output_device() -> Result<Device> {
    cpal::default_host().default_output_device()
        .ok_or_else(|| AudioError::NoOutputDevice.into())
}

pub fn start_output(device: &Device, sample_rate: u32, rx: Receiver<Vec<f32>>) -> Result<Stream> {
    let config = StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };
    let buf = std::sync::Arc::new(parking_lot::Mutex::new(
        std::collections::VecDeque::<f32>::with_capacity(4096),
    ));
    let bp = buf.clone();
    std::thread::Builder::new().name("audio-feeder".into()).spawn(move || {
        while let Ok(frame) = rx.recv() {
            let mut b = bp.lock();
            b.extend(frame);
            while b.len() > 16_000 { b.pop_front(); }
        }
    })?;
    let stream = device.build_output_stream(
        &config,
        move |data: &mut [f32], _| {
            let mut b = buf.lock();
            for s in data.iter_mut() { *s = b.pop_front().unwrap_or(0.0); }
        },
        |err| log::error!("Output error: {}", err),
        None,
    )?;
    stream.play().context("failed to start output")?;
    Ok(stream)
}
