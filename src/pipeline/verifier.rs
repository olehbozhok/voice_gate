//! Background speaker verification — runs ECAPA-TDNN in a separate thread.
//!
//! The main processor thread sends audio windows via a channel.
//! This thread computes embeddings and updates the shared verification result.
//! The processor never blocks on embedding extraction.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{Sender, bounded};
use parking_lot::RwLock;

use crate::config::Config;
use crate::speaker::cosine_similarity;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::VoiceProfile;

/// Shared verification result, updated by the verifier thread,
/// read by the processor thread.
#[derive(Debug, Clone, Copy)]
pub struct VerificationResult {
    /// Cosine similarity between the speaker and enrolled profile.
    pub similarity: f32,
    /// Whether similarity >= threshold at the time of computation.
    pub is_owner: bool,
}

/// Handle to the background verifier thread.
/// Send audio windows via `submit()`, read results via `result()`.
pub struct SpeakerVerifier {
    tx: Sender<Vec<f32>>,
    result: Arc<RwLock<Option<VerificationResult>>>,
    /// Set to true once at least one verification has completed.
    verified: Arc<AtomicBool>,
    /// Whether an enrolled voice profile was provided.
    has_profile: bool,
}

impl SpeakerVerifier {
    /// Spawn the background verifier thread.
    ///
    /// * `ecapa` — the ECAPA-TDNN model (moved into the thread).
    /// * `profile` — the enrolled voice profile to compare against.
    /// * `config` — shared config, read each verification for current threshold.
    pub fn spawn(
        mut ecapa: EcapaTdnn,
        profile: Option<VoiceProfile>,
        config: Arc<RwLock<Config>>,
    ) -> Self {
        /// Maximum number of pending audio windows in the channel.
        const CHANNEL_CAPACITY: usize = 4;

        let has_profile = profile.is_some();
        let (tx, rx) = bounded::<Vec<f32>>(CHANNEL_CAPACITY);
        let result: Arc<RwLock<Option<VerificationResult>>> = Arc::new(RwLock::new(None));
        let verified = Arc::new(AtomicBool::new(false));

        let result_writer = result.clone();
        let verified_flag = verified.clone();

        std::thread::Builder::new()
            .name("speaker-verifier".into())
            .spawn(move || {
                log::info!("Speaker verifier thread started");
                while let Ok(window) = rx.recv() {
                    let profile = match &profile {
                        Some(p) => p,
                        None => continue,
                    };

                    match ecapa.extract(&window) {
                        Ok(embedding) => {
                            let threshold = config.read().speaker.similarity_threshold;
                            let sim = cosine_similarity(&profile.centroid, &embedding);
                            let is_owner = sim >= threshold;
                            log::trace!("Speaker similarity: {:.3} (owner: {})", sim, is_owner);
                            *result_writer.write() = Some(VerificationResult {
                                similarity: sim,
                                is_owner,
                            });
                            verified_flag.store(true, Ordering::Relaxed);
                        }
                        Err(e) => {
                            log::warn!("Embedding extraction failed: {}", e);
                        }
                    }
                }
                log::info!("Speaker verifier thread stopped");
            })
            .expect("failed to spawn speaker-verifier thread");

        Self { tx, result, verified, has_profile }
    }

    /// Submit an audio window for background verification.
    /// Non-blocking — drops the request if the channel is full.
    pub fn submit(&self, window: Vec<f32>) {
        let _ = self.tx.try_send(window);
    }

    /// Read the latest verification result (non-blocking).
    pub fn result(&self) -> Option<VerificationResult> {
        *self.result.read()
    }

    /// Whether at least one verification has completed.
    pub fn has_verified(&self) -> bool {
        self.verified.load(Ordering::Relaxed)
    }

    /// Whether an enrolled voice profile was provided.
    pub fn has_profile(&self) -> bool {
        self.has_profile
    }

    /// Reset the verification state (e.g. after long silence).
    pub fn reset(&self) {
        *self.result.write() = None;
        self.verified.store(false, Ordering::Relaxed);
    }
}
