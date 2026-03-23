//! Audio processor — orchestrates VAD -> Speaker Verification -> Gate.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

use super::recorder::TestRecorder;
use super::state_machine::GateState;
use super::verifier::SpeakerVerifier;
use crate::audio::AudioFrame;
use crate::config::{Config, GateInput};
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::enrollment::{EnrollmentSession, EnrollmentState};
use crate::speaker::profile::{ProfileStore, VoiceProfile};
use crate::vad::silero::SileroVad;
use crate::vad::VadResult;

/// Duration of the speaker verification sliding window in seconds.
const VERIFICATION_WINDOW_SECS: f32 = 1.5;

/// Commands sent from the UI thread to control enrollment on the processor thread.
pub enum EnrollmentCommand {
    /// Start recording speech for a new voice profile.
    Start,
    /// Finish recording and build the voice profile.
    Finalize,
    /// Cancel enrollment and discard recorded audio.
    Cancel,
}

/// Telemetry snapshot shared with the UI thread.
#[derive(Debug, Clone)]
pub struct PipelineTelemetry {
    pub gate_state: GateState,
    pub input_level: f32,
    pub vad_probability: f32,
    pub speaker_similarity: f32,
    /// Name of the best-matching voice profile.
    pub matched_profile: Option<String>,
    pub gate_open: bool,
    /// Current enrollment state, if enrollment is active.
    pub enrollment_state: EnrollmentState,
    /// Accumulated speech seconds during enrollment.
    pub enrollment_speech_secs: f32,
}

impl Default for PipelineTelemetry {
    fn default() -> Self {
        Self {
            gate_state: GateState::Silent,
            input_level: 0.0,
            vad_probability: 0.0,
            speaker_similarity: 0.0,
            matched_profile: None,
            gate_open: false,
            enrollment_state: EnrollmentState::Idle,
            enrollment_speech_secs: 0.0,
        }
    }
}

/// Main audio processor.
///
/// Runs VAD on every frame (32ms). Speaker verification runs in a
/// separate thread via [`SpeakerVerifier`] — the processor sends audio
/// windows and reads results without blocking.
pub struct Processor {
    config: Arc<RwLock<Config>>,
    vad: SileroVad,
    verifier: SpeakerVerifier,
    /// Continuous silence duration in milliseconds (reset on speech).
    silence_ms: u32,
    /// Duration of a single frame in milliseconds.
    frame_ms: u32,
    /// Input device channel count.
    input_channels: u16,
    /// Input device sample rate.
    input_rate: u32,
    /// Output device channel count.
    output_channels: u16,
    /// Output device sample rate.
    output_rate: u32,
    verification_buffer: VecDeque<f32>,
    verification_window_samples: usize,
    /// Delay line storing original-quality audio. Audio enters at the back
    /// and is taken from the front when the gate is open, adding a constant
    /// `pre_buffer_ms` latency that preserves word onsets.
    pre_buffer: VecDeque<f32>,
    /// Whether the gate was open on the previous frame.
    prev_gate_open: bool,
    /// How long similarity has been available (ms since first verification).
    similarity_available_ms: u32,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    recording_flag: Arc<AtomicBool>,
    recorder: Option<TestRecorder>,
    enrollment: Option<EnrollmentSession>,
    enrollment_rx: Receiver<EnrollmentCommand>,
    /// Separate ECAPA-TDNN instance for enrollment (main one is in verifier thread).
    enrollment_ecapa: EcapaTdnn,
    /// Shared profile store for saving new enrollments.
    profile_store: Arc<RwLock<ProfileStore>>,
}

impl Processor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<RwLock<Config>>,
        vad: SileroVad,
        verifier: SpeakerVerifier,
        enrollment_ecapa: EcapaTdnn,
        telemetry: Arc<RwLock<PipelineTelemetry>>,
        recording_flag: Arc<AtomicBool>,
        enrollment_rx: Receiver<EnrollmentCommand>,
        profile_store: Arc<RwLock<ProfileStore>>,
        input_channels: u16,
        input_rate: u32,
        output_channels: u16,
        output_rate: u32,
    ) -> Self {
        let cfg = config.read();
        let verification_window_samples =
            (VERIFICATION_WINDOW_SECS * cfg.audio.sample_rate as f32) as usize;
        let frame_ms =
            (cfg.audio.frame_samples as f32 / cfg.audio.sample_rate as f32 * 1000.0) as u32;
        drop(cfg);
        Self {
            config,
            vad,
            verifier,
            silence_ms: 0,
            frame_ms,
            input_channels,
            input_rate,
            output_channels,
            output_rate,
            enrollment_ecapa,
            verification_buffer: VecDeque::with_capacity(verification_window_samples),
            verification_window_samples,
            pre_buffer: VecDeque::new(),
            prev_gate_open: false,
            similarity_available_ms: 0,
            telemetry,
            recording_flag,
            recorder: None,
            enrollment: None,
            enrollment_rx,
            profile_store,
        }
    }

    /// Run the processing loop. Blocks until `rx_input` is closed.
    pub fn run(
        &mut self,
        rx_input: Receiver<AudioFrame>,
        tx_output: Sender<Vec<f32>>,
    ) -> Result<()> {
        log::info!("Processor started (ort, ONNX Runtime)");
        while let Ok(audio_frame) = rx_input.recv() {
            self.handle_enrollment_commands();
            let output = self.process_frame(&audio_frame)?;
            let _ = tx_output.try_send(output);
        }
        log::info!("Processor stopping");
        Ok(())
    }

    fn process_frame(&mut self, audio_frame: &AudioFrame) -> Result<Vec<f32>> {
        let frame = &audio_frame.pipeline;
        let input_level = crate::audio::rms(frame);

        // Stage 1: VAD
        let speech_probability = self.vad.process(frame)?;

        // Stage 2: Speaker verification — accumulate only voiced frames.
        let vad_threshold = self.config.read().vad.threshold;
        let is_speech = speech_probability >= vad_threshold;
        if is_speech {
            self.silence_ms = 0;
            self.verification_buffer.extend(frame.iter());
        } else {
            self.silence_ms = self.silence_ms.saturating_add(self.frame_ms);
        }
        self.submit_verification_window();

        // Stage 3: GateMode makes the full decision.
        let ver_result = self.verifier.result();
        let similarity = ver_result.as_ref().map(|r| r.similarity).unwrap_or(0.0);
        let matched_profile = ver_result.as_ref().and_then(|r| r.matched_profile.clone());
        // Track how long similarity has been available.
        if self.verifier.has_verified() {
            self.similarity_available_ms =
                self.similarity_available_ms.saturating_add(self.frame_ms);
        }
        let cfg = self.config.read();
        let gate_input = GateInput {
            speech_probability,
            vad_threshold,
            verified: self.verifier.has_verified(),
            similarity: ver_result.as_ref().map(|r| r.similarity),
            similarity_threshold: cfg.speaker.similarity_threshold,
            has_profile: self.verifier.has_profile(),
            hold_time_ms: cfg.gate.hold_time_ms,
            silence_ms: self.silence_ms,
            similarity_available_ms: self.similarity_available_ms,
        };
        let decision = cfg.gate.mode.evaluate(&gate_input);
        drop(cfg);

        if decision.flush_verification {
            self.verification_buffer.clear();
            self.verifier.reset();
            self.similarity_available_ms = 0;
        }

        let gate_open = decision.pass_audio;
        let vad_result = VadResult {
            speech_probability,
            is_speech,
        };

        // Derive gate state for telemetry display.
        let state = if is_speech && gate_open {
            GateState::MyVoice
        } else if is_speech && !gate_open {
            GateState::OtherVoice
        } else if gate_open {
            GateState::Trailing
        } else {
            GateState::Silent
        };

        // Update telemetry for UI
        {
            let mut t = self.telemetry.write();
            t.gate_state = state;
            t.input_level = input_level;
            t.vad_probability = vad_result.speech_probability;
            t.speaker_similarity = similarity;
            t.matched_profile = matched_profile;
            t.gate_open = gate_open;
        }

        // Enrollment: feed audio frames when recording
        if let Some(ref mut session) = self.enrollment {
            session.feed_frame(frame, &vad_result);
            let secs = session.speech_seconds();
            log::trace!(
                "Enrollment: vad={:.3} is_speech={} speech_secs={:.1} state={:?}",
                vad_result.speech_probability,
                vad_result.is_speech,
                secs,
                session.state
            );
            let mut t = self.telemetry.write();
            t.enrollment_state = session.state.clone();
            t.enrollment_speech_secs = secs;
        }

        // Pre-buffer: delay line that preserves word onsets.
        // Audio enters at back, exits from front with pre_buffer_ms delay.
        let pre_buffer_ms = self.config.read().gate.pre_buffer_ms;
        let frame_size = audio_frame.original.len();
        let pre_buffer_cap =
            ((pre_buffer_ms as f64 / 1000.0 * self.input_rate as f64 * self.input_channels as f64)
                .round() as usize)
                .max(frame_size);
        // add current frame
        self.pre_buffer.extend(audio_frame.original.iter());
        // Drop old audio from the front if pre-buffer exceeds capacity
        // or pre-buffer capacity is changed.
        while self.pre_buffer.len() > pre_buffer_cap + frame_size {
            self.pre_buffer.pop_front();
        }

        // Return audio if gate is open
        let output = if gate_open {
            let delayed: Vec<f32> = self.pre_buffer.drain(..frame_size).collect();
            self.convert_to_output(&delayed)
        } else {
            vec![0.0; self.output_frame_size(frame_size)]
        };
        self.prev_gate_open = gate_open;

        // Recording
        let should_record = self.recording_flag.load(Ordering::Relaxed);
        self.update_recording(should_record, frame, &output);

        Ok(output)
    }

    /// Calculate output frame size from an input frame size.
    fn output_frame_size(&self, input_frame_size: usize) -> usize {
        (input_frame_size as f64 / (self.input_rate as f64 * self.input_channels as f64)
            * self.output_rate as f64
            * self.output_channels as f64)
            .round() as usize
    }

    /// Convert interleaved audio from input device format to output device format.
    /// Handles channel count and sample rate differences.
    fn convert_to_output(&self, input: &[f32]) -> Vec<f32> {
        let mut samples = input.to_vec();

        // Channel conversion: input channels → output channels.
        if self.input_channels != self.output_channels {
            let mono = crate::audio::channels_to_mono(&samples, self.input_channels);
            samples = crate::audio::mono_to_channels(&mono, self.output_channels);
        }

        // Sample rate conversion if devices differ.
        if self.input_rate != self.output_rate {
            // Deinterleave, resample each channel, reinterleave.
            let ch = self.output_channels as usize;
            let mut channels: Vec<Vec<f32>> = (0..ch)
                .map(|c| samples.iter().skip(c).step_by(ch).copied().collect())
                .collect();
            for channel in &mut channels {
                *channel =
                    crate::audio::resampler::resample(channel, self.input_rate, self.output_rate);
            }
            let len = channels[0].len();
            samples = Vec::with_capacity(len * ch);
            for i in 0..len {
                for channel in &channels {
                    samples.push(channel[i]);
                }
            }
        }

        samples
    }

    /// Accumulate audio and submit to the background verifier when
    /// enough data is available. Non-blocking — the verifier thread
    /// computes the embedding asynchronously.
    /// Submit a verification window to the background thread if enough
    /// audio has accumulated. Called every frame regardless of VAD state.
    fn submit_verification_window(&mut self) {
        if self.verification_buffer.len() < self.verification_window_samples {
            return;
        }

        let window: Vec<f32> = self.verification_buffer.iter().copied().collect();
        let drain_count = self.verification_window_samples / 2;
        self.verification_buffer.drain(..drain_count);

        self.verifier.submit(window);
    }

    /// Process any pending enrollment commands from the UI thread.
    fn handle_enrollment_commands(&mut self) {
        while let Ok(cmd) = self.enrollment_rx.try_recv() {
            match cmd {
                EnrollmentCommand::Start => {
                    log::info!("Enrollment started");
                    let cfg = self.config.read();
                    let mut session = EnrollmentSession::new(cfg.audio.sample_rate);
                    session.start();
                    self.enrollment = Some(session);
                    let mut t = self.telemetry.write();
                    t.enrollment_state = EnrollmentState::Recording {
                        speech_seconds: 0.0,
                    };
                    t.enrollment_speech_secs = 0.0;
                }
                EnrollmentCommand::Finalize => {
                    self.finalize_enrollment();
                }
                EnrollmentCommand::Cancel => {
                    log::info!("Enrollment cancelled");
                    self.enrollment = None;
                    let mut t = self.telemetry.write();
                    t.enrollment_state = EnrollmentState::Idle;
                    t.enrollment_speech_secs = 0.0;
                }
            }
        }
    }

    /// Extract embeddings from recorded speech and save voice profile.
    fn finalize_enrollment(&mut self) {
        let session = match self.enrollment.take() {
            Some(s) => s,
            None => return,
        };

        {
            let mut t = self.telemetry.write();
            t.enrollment_state = EnrollmentState::Processing;
        }

        let windows = session.get_embedding_windows();
        if windows.is_empty() {
            log::warn!("Enrollment: no valid speech windows");
            let mut t = self.telemetry.write();
            t.enrollment_state =
                EnrollmentState::Failed("No valid speech segments recorded".into());
            return;
        }

        log::info!(
            "Enrollment: extracting embeddings from {} windows",
            windows.len()
        );
        let mut embeddings = Vec::new();
        for window in &windows {
            match self.enrollment_ecapa.extract(window) {
                Ok(emb) => embeddings.push(emb),
                Err(e) => {
                    log::warn!("Embedding extraction failed for a window: {}", e);
                }
            }
        }

        if embeddings.is_empty() {
            let mut t = self.telemetry.write();
            t.enrollment_state = EnrollmentState::Failed("Failed to extract any embeddings".into());
            return;
        }

        let duration = session.speech_seconds();
        let count = self.profile_store.read().len() + 1;
        let name = format!("Profile {}", count);
        match VoiceProfile::from_embeddings(&name, &embeddings, duration) {
            Ok(profile) => match self.profile_store.write().add(profile) {
                Ok(()) => {
                    log::info!(
                        "Enrollment complete: '{}', {} segments, {:.1}s",
                        name,
                        embeddings.len(),
                        duration
                    );
                    let mut t = self.telemetry.write();
                    t.enrollment_state = EnrollmentState::Done;
                }
                Err(e) => {
                    log::error!("Failed to save profile: {}", e);
                    let mut t = self.telemetry.write();
                    t.enrollment_state = EnrollmentState::Failed(format!("Save failed: {}", e));
                }
            },
            Err(e) => {
                log::error!("Failed to build profile: {}", e);
                let mut t = self.telemetry.write();
                t.enrollment_state = EnrollmentState::Failed(format!("Build failed: {}", e));
            }
        }
    }

    /// Handle recording state transitions and write frames.
    fn update_recording(&mut self, should_record: bool, original: &[f32], gated: &[f32]) {
        match (self.recorder.is_some(), should_record) {
            // Start recording
            (false, true) => match TestRecorder::new() {
                Ok(rec) => self.recorder = Some(rec),
                Err(e) => {
                    log::error!("Failed to start recording: {}", e);
                    self.recording_flag.store(false, Ordering::Relaxed);
                }
            },
            // Stop recording
            (true, false) => {
                if let Some(rec) = self.recorder.take() {
                    if let Err(e) = rec.finish() {
                        log::error!("Failed to finalize recording: {}", e);
                    }
                }
            }
            _ => {}
        }

        // Write frames if recording
        if let Some(rec) = self.recorder.as_mut() {
            let write_err = rec
                .write_original(original)
                .and_then(|()| rec.write_gated(gated))
                .err();
            if let Some(e) = write_err {
                log::warn!("Recording write error: {}", e);
                self.recorder = None;
                self.recording_flag.store(false, Ordering::Relaxed);
            }
        }
    }
}
