//! Audio processor — orchestrates VAD -> Speaker Verification -> Gate.

use std::collections::VecDeque;
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::speaker::cosine_similarity;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::VoiceProfile;
use crate::vad::silero::SileroVad;
use super::state_machine::{GateState, GateStateMachine};

/// Telemetry snapshot shared with the UI thread.
#[derive(Debug, Clone)]
pub struct PipelineTelemetry {
    pub gate_state: GateState,
    pub input_level: f32,
    pub vad_probability: f32,
    pub speaker_similarity: f32,
    pub gate_open: bool,
}

impl Default for PipelineTelemetry {
    fn default() -> Self {
        Self { gate_state: GateState::Silent, input_level: 0.0, vad_probability: 0.0, speaker_similarity: 0.0, gate_open: false }
    }
}

/// Main audio processor.
pub struct Processor {
    config: Config,
    vad: SileroVad,
    ecapa: EcapaTdnn,
    profile: Option<VoiceProfile>,
    gate: GateStateMachine,
    verification_buffer: VecDeque<f32>,
    verification_window_samples: usize,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
}

impl Processor {
    pub fn new(
        config: Config,
        vad: SileroVad,
        ecapa: EcapaTdnn,
        profile: Option<VoiceProfile>,
        telemetry: Arc<RwLock<PipelineTelemetry>>,
    ) -> Self {
        let verification_window_samples = (1.5 * config.audio.sample_rate as f32) as usize;
        Self {
            gate: GateStateMachine::new(config.gate.hold_time_ms),
            config, vad, ecapa, profile,
            verification_buffer: VecDeque::with_capacity(verification_window_samples),
            verification_window_samples,
            telemetry,
        }
    }

    /// Run the processing loop. Blocks until `rx_input` is closed.
    pub fn run(
        &mut self,
        rx_input: Receiver<Vec<f32>>,
        tx_output: Sender<Vec<f32>>,
    ) -> Result<()> {
        log::info!("Processor started (tract, CPU)");
        while let Ok(frame) = rx_input.recv() {
            let output = self.process_frame(&frame)?;
            let _ = tx_output.try_send(output);
        }
        log::info!("Processor stopping");
        Ok(())
    }

    fn process_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let input_level = crate::audio::rms(frame);

        // Stage 1: VAD
        let vad_result = self.vad.process(frame)?;

        // Stage 2: Speaker verification
        let (is_owner, similarity) = if vad_result.is_speech {
            self.run_speaker_verification(frame)
        } else {
            self.verification_buffer.clear();
            (false, 0.0)
        };

        // Stage 3: Gate
        let state = self.gate.update(vad_result.is_speech, is_owner);

        // Update telemetry for UI
        {
            let mut t = self.telemetry.write();
            t.gate_state = state;
            t.input_level = input_level;
            t.vad_probability = vad_result.speech_probability;
            t.speaker_similarity = similarity;
            t.gate_open = state.is_open();
        }

        if state.is_open() {
            Ok(frame.to_vec())
        } else {
            Ok(vec![0.0; frame.len()])
        }
    }

    fn run_speaker_verification(&mut self, frame: &[f32]) -> (bool, f32) {
        self.verification_buffer.extend(frame.iter());

        if self.verification_buffer.len() < self.verification_window_samples {
            let prev_sim = self.telemetry.read().speaker_similarity;
            let threshold = self.config.speaker.similarity_threshold;
            return (prev_sim >= threshold, prev_sim);
        }

        let window: Vec<f32> = self.verification_buffer.iter().copied().collect();
        let drain_count = self.verification_window_samples / 2;
        self.verification_buffer.drain(..drain_count);

        let profile = match &self.profile {
            Some(p) => p,
            None => return (true, 1.0),
        };

        match self.ecapa.extract(&window) {
            Ok(embedding) => {
                let sim = cosine_similarity(&profile.centroid, &embedding);
                let is_owner = sim >= self.config.speaker.similarity_threshold;
                log::trace!("Speaker similarity: {:.3} (owner: {})", sim, is_owner);
                (is_owner, sim)
            }
            Err(e) => {
                log::warn!("Embedding extraction failed: {}", e);
                (false, 0.0)
            }
        }
    }

    pub fn set_profile(&mut self, profile: VoiceProfile) {
        log::info!("Profile updated: '{}'", profile.name);
        self.profile = Some(profile);
    }

    pub fn update_config(&mut self, config: &Config) {
        self.vad.set_threshold(config.vad.threshold);
        self.gate.set_hold_time(config.gate.hold_time_ms);
        self.config.speaker.similarity_threshold = config.speaker.similarity_threshold;
    }
}
