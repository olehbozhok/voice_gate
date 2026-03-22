//! Background speaker verification — runs ECAPA-TDNN in a separate thread.
//!
//! The main processor thread sends audio windows via a channel.
//! This thread computes embeddings and compares against ALL enrolled
//! profiles, taking the maximum similarity. The processor never blocks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Sender};
use parking_lot::RwLock;

use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::ProfileStore;

/// Shared verification result, updated by the verifier thread.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Best cosine similarity across all enrolled profiles.
    pub similarity: f32,
    /// Name of the profile with the best match.
    pub matched_profile: Option<String>,
}

/// Handle to the background verifier thread.
pub struct SpeakerVerifier {
    tx: Sender<Vec<f32>>,
    result: Arc<RwLock<Option<VerificationResult>>>,
    verified: Arc<AtomicBool>,
    profile_store: Arc<RwLock<ProfileStore>>,
}

impl SpeakerVerifier {
    /// Spawn the background verifier thread.
    ///
    /// * `ecapa` — the ECAPA-TDNN model (moved into the thread).
    /// * `profile_store` — shared profile store, read each verification cycle.
    pub fn spawn(mut ecapa: EcapaTdnn, profile_store: Arc<RwLock<ProfileStore>>) -> Self {
        /// Maximum pending audio windows in the channel.
        const CHANNEL_CAPACITY: usize = 4;

        let (tx, rx) = bounded::<Vec<f32>>(CHANNEL_CAPACITY);
        let result: Arc<RwLock<Option<VerificationResult>>> = Arc::new(RwLock::new(None));
        let verified = Arc::new(AtomicBool::new(false));

        let result_writer = result.clone();
        let verified_flag = verified.clone();
        let store = profile_store.clone();

        std::thread::Builder::new()
            .name("speaker-verifier".into())
            .spawn(move || {
                log::info!("Speaker verifier thread started");
                while let Ok(window) = rx.recv() {
                    let profiles = store.read();
                    if profiles.is_empty() {
                        drop(profiles);
                        continue;
                    }

                    match ecapa.extract(&window) {
                        Ok(embedding) => {
                            let mut best_sim = 0.0f32;
                            let mut best_name: Option<String> = None;
                            for profile in profiles.profiles() {
                                let sim = profile.similarity(&embedding);
                                if sim > best_sim {
                                    best_sim = sim;
                                    best_name = Some(profile.name.clone());
                                }
                            }
                            let count = profiles.len();
                            drop(profiles);

                            log::trace!(
                                "Speaker similarity: {:.3} (best of {}, matched: {:?})",
                                best_sim,
                                count,
                                best_name
                            );

                            *result_writer.write() = Some(VerificationResult {
                                similarity: best_sim,
                                matched_profile: best_name,
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

        Self {
            tx,
            result,
            verified,
            profile_store,
        }
    }

    /// Submit an audio window for background verification (non-blocking).
    pub fn submit(&self, window: Vec<f32>) {
        let _ = self.tx.try_send(window);
    }

    /// Read the latest verification result (non-blocking).
    pub fn result(&self) -> Option<VerificationResult> {
        self.result.read().clone()
    }

    /// Whether at least one verification has completed.
    pub fn has_verified(&self) -> bool {
        self.verified.load(Ordering::Relaxed)
    }

    /// Whether any enrolled profiles exist.
    pub fn has_profile(&self) -> bool {
        !self.profile_store.read().is_empty()
    }

    /// Reset the verification state.
    pub fn reset(&self) {
        *self.result.write() = None;
        self.verified.store(false, Ordering::Relaxed);
    }
}
