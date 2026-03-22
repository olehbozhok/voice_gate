//! Background speaker verification — runs ECAPA-TDNN in a separate thread.
//!
//! The main processor thread sends audio windows via a channel.
//! This thread computes embeddings and compares against ALL enrolled
//! profiles, taking the maximum similarity. The processor never blocks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{Sender, bounded};
use parking_lot::RwLock;

use crate::config::Config;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::VoiceProfile;

/// Shared verification result, updated by the verifier thread.
#[derive(Debug, Clone, Copy)]
pub struct VerificationResult {
    /// Best cosine similarity across all enrolled profiles.
    pub similarity: f32,
    /// Whether best similarity >= threshold.
    pub is_owner: bool,
}

/// Handle to the background verifier thread.
pub struct SpeakerVerifier {
    tx: Sender<Vec<f32>>,
    result: Arc<RwLock<Option<VerificationResult>>>,
    verified: Arc<AtomicBool>,
    has_profiles: bool,
}

impl SpeakerVerifier {
    /// Spawn the background verifier thread.
    ///
    /// * `ecapa` — the ECAPA-TDNN model (moved into the thread).
    /// * `profiles` — all enrolled voice profiles to compare against.
    /// * `config` — shared config for live threshold reading.
    pub fn spawn(
        mut ecapa: EcapaTdnn,
        profiles: Vec<VoiceProfile>,
        config: Arc<RwLock<Config>>,
    ) -> Self {
        /// Maximum pending audio windows in the channel.
        const CHANNEL_CAPACITY: usize = 4;

        let has_profiles = !profiles.is_empty();
        let (tx, rx) = bounded::<Vec<f32>>(CHANNEL_CAPACITY);
        let result: Arc<RwLock<Option<VerificationResult>>> = Arc::new(RwLock::new(None));
        let verified = Arc::new(AtomicBool::new(false));

        let result_writer = result.clone();
        let verified_flag = verified.clone();

        std::thread::Builder::new()
            .name("speaker-verifier".into())
            .spawn(move || {
                log::info!("Speaker verifier thread started ({} profiles)", profiles.len());
                while let Ok(window) = rx.recv() {
                    if profiles.is_empty() { continue; }

                    match ecapa.extract(&window) {
                        Ok(embedding) => {
                            let threshold = config.read().speaker.similarity_threshold;

                            // Compare against all profiles, take best match.
                            let mut best_sim = 0.0f32;
                            for profile in &profiles {
                                let sim = profile.similarity(&embedding);
                                best_sim = best_sim.max(sim);
                            }

                            let is_owner = best_sim >= threshold;
                            log::trace!("Speaker similarity: {:.3} (best of {}, owner: {})",
                                best_sim, profiles.len(), is_owner);

                            *result_writer.write() = Some(VerificationResult {
                                similarity: best_sim,
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

        Self { tx, result, verified, has_profiles }
    }

    /// Submit an audio window for background verification (non-blocking).
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

    /// Whether any enrolled profiles were provided.
    pub fn has_profile(&self) -> bool {
        self.has_profiles
    }

    /// Reset the verification state.
    pub fn reset(&self) {
        *self.result.write() = None;
        self.verified.store(false, Ordering::Relaxed);
    }
}
