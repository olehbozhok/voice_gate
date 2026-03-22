//! Audio processor — orchestrates VAD -> Speaker Verification -> Gate.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

use super::recorder::TestRecorder;
use super::state_machine::{GateState, GateStateMachine};
use super::verifier::SpeakerVerifier;
use crate::config::{Config, VerificationState};
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::enrollment::{EnrollmentSession, EnrollmentState};
use crate::speaker::profile::VoiceProfile;
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
    gate: GateStateMachine,
    verifier: SpeakerVerifier,
    verification_buffer: VecDeque<f32>,
    verification_window_samples: usize,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    recording_flag: Arc<AtomicBool>,
    recorder: Option<TestRecorder>,
    enrollment: Option<EnrollmentSession>,
    enrollment_rx: Receiver<EnrollmentCommand>,
    /// Separate ECAPA-TDNN instance for enrollment (main one is in verifier thread).
    enrollment_ecapa: EcapaTdnn,
    profile_dir: std::path::PathBuf,
}

impl Processor {
    pub fn new(
        config: Arc<RwLock<Config>>,
        vad: SileroVad,
        verifier: SpeakerVerifier,
        enrollment_ecapa: EcapaTdnn,
        telemetry: Arc<RwLock<PipelineTelemetry>>,
        recording_flag: Arc<AtomicBool>,
        enrollment_rx: Receiver<EnrollmentCommand>,
        profile_dir: std::path::PathBuf,
    ) -> Self {
        let cfg = config.read();
        let verification_window_samples =
            (VERIFICATION_WINDOW_SECS * cfg.audio.sample_rate as f32) as usize;
        let gate = GateStateMachine::new(cfg.gate.hold_time_ms);
        drop(cfg);
        Self {
            gate,
            config,
            vad,
            verifier,
            enrollment_ecapa,
            verification_buffer: VecDeque::with_capacity(verification_window_samples),
            verification_window_samples,
            telemetry,
            recording_flag,
            recorder: None,
            enrollment: None,
            enrollment_rx,
            profile_dir,
        }
    }

    /// Run the processing loop. Blocks until `rx_input` is closed.
    pub fn run(&mut self, rx_input: Receiver<Vec<f32>>, tx_output: Sender<Vec<f32>>) -> Result<()> {
        log::info!("Processor started (ort, ONNX Runtime)");
        while let Ok(frame) = rx_input.recv() {
            self.handle_enrollment_commands();
            let output = self.process_frame(&frame)?;
            let _ = tx_output.try_send(output);
        }
        log::info!("Processor stopping");
        Ok(())
    }

    fn process_frame(&mut self, frame: &[f32]) -> Result<Vec<f32>> {
        let input_level = crate::audio::rms(frame);

        // Read live config (may change at runtime via Settings UI).
        let cfg = self.config.read();
        self.gate.set_hold_time(cfg.gate.hold_time_ms);
        let vad_threshold = cfg.vad.threshold;
        drop(cfg);

        // Stage 1: VAD
        let speech_probability = self.vad.process(frame)?;
        let vad_result = VadResult {
            speech_probability,
            is_speech: speech_probability >= vad_threshold,
        };

        // Stage 2: Speaker verification — accumulate only voiced frames.
        // Buffer is never cleared — sliding window (50% drain) naturally
        // flushes old audio. Silence frames are skipped to keep the buffer
        // dense with speech, improving embedding quality.
        if vad_result.is_speech {
            self.verification_buffer.extend(frame.iter());
        }
        self.submit_verification_window();

        // Stage 3: Gate mode decides open/closed (reads latest result from verifier)
        let ver_result = self.verifier.result();
        let cfg = self.config.read();
        let verification = VerificationState {
            is_speech: vad_result.is_speech,
            verified: self.verifier.has_verified(),
            similarity: ver_result.map(|r| r.similarity),
            threshold: cfg.speaker.similarity_threshold,
            has_profile: self.verifier.has_profile(),
        };
        let is_owner = cfg.gate.mode.should_open(&verification);
        drop(cfg);
        let similarity = ver_result.map(|r| r.similarity).unwrap_or(0.0);

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

        let output = if state.is_open() {
            frame.to_vec()
        } else {
            vec![0.0; frame.len()]
        };

        // Recording
        let should_record = self.recording_flag.load(Ordering::Relaxed);
        self.update_recording(should_record, frame, &output);

        Ok(output)
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
                    let mut session = EnrollmentSession::new(
                        cfg.audio.sample_rate,
                        cfg.speaker.min_enrollment_seconds,
                    );
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
        match VoiceProfile::from_embeddings("default", &embeddings, duration) {
            Ok(profile) => {
                let path = self.profile_dir.join("default.json");
                match profile.save(&path) {
                    Ok(()) => {
                        log::info!(
                            "Enrollment complete: {} segments, {:.1}s",
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
                }
            }
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
