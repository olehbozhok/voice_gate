//! WAV recording of original (pre-gate) and gated (post-gate) audio for testing.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use hound::{SampleFormat, WavSpec, WavWriter};

use crate::audio::PIPELINE_SAMPLE_RATE;

/// Number of audio channels in recorded WAV files (mono).
const WAV_CHANNELS: u16 = 1;

/// Bits per sample in recorded WAV files (32-bit float).
const WAV_BITS_PER_SAMPLE: u16 = 32;

/// Records original (pre-gate) and gated (post-gate) audio to WAV files.
pub struct TestRecorder {
    original: WavWriter<BufWriter<File>>,
    gated: WavWriter<BufWriter<File>>,
}

impl TestRecorder {
    /// Create a new recorder writing to `test_original.wav` and `test_gated.wav`
    /// in the current working directory.
    pub fn new() -> Result<Self> {
        let spec = WavSpec {
            channels: WAV_CHANNELS,
            sample_rate: PIPELINE_SAMPLE_RATE,
            bits_per_sample: WAV_BITS_PER_SAMPLE,
            sample_format: SampleFormat::Float,
        };

        let original = WavWriter::create(Path::new("test_original.wav"), spec)
            .context("failed to create test_original.wav")?;
        let gated = WavWriter::create(Path::new("test_gated.wav"), spec)
            .context("failed to create test_gated.wav")?;

        log::info!("Test recording started: test_original.wav + test_gated.wav");
        Ok(Self { original, gated })
    }

    /// Write a frame of original (pre-gate) audio.
    pub fn write_original(&mut self, frame: &[f32]) -> Result<()> {
        for &s in frame {
            self.original.write_sample(s)?;
        }
        Ok(())
    }

    /// Write a frame of gated (post-gate) audio.
    pub fn write_gated(&mut self, frame: &[f32]) -> Result<()> {
        for &s in frame {
            self.gated.write_sample(s)?;
        }
        Ok(())
    }

    /// Finalize and close both WAV files.
    pub fn finish(self) -> Result<()> {
        self.original.finalize().context("failed to finalize test_original.wav")?;
        self.gated.finalize().context("failed to finalize test_gated.wav")?;
        log::info!("Test recording finished");
        Ok(())
    }
}

impl Drop for TestRecorder {
    fn drop(&mut self) {
        // Safety net: hound::WavWriter flushes on drop, but silently.
        // Prefer calling finish() explicitly for proper error handling.
    }
}
