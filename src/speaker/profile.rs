//! Voice profiles — stored speaker embedding centroids.
//!
//! Multiple profiles can be enrolled. During verification, the best
//! matching profile (highest cosine similarity) is used.

use super::cosine_similarity;
use crate::error::ProfileError;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single enrolled voice profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    /// Display name (editable by user).
    pub name: String,
    /// L2-normalized embedding centroid (192 dimensions).
    pub centroid: Vec<f32>,
    /// Number of speech segments used to build the centroid.
    pub num_segments: usize,
    /// Total speech duration used for enrollment (seconds).
    pub total_duration_secs: f32,
    /// ISO 8601 timestamp when the profile was created.
    pub created_at: String,
}

impl VoiceProfile {
    /// Build a profile by averaging multiple embeddings into a single centroid.
    pub fn from_embeddings(
        name: impl Into<String>,
        embeddings: &[Vec<f32>],
        duration: f32,
    ) -> Result<Self> {
        if embeddings.is_empty() {
            anyhow::bail!("zero embeddings");
        }
        let dim = embeddings[0].len();
        let mut centroid = vec![0.0f32; dim];
        for emb in embeddings {
            for (c, e) in centroid.iter_mut().zip(emb.iter()) {
                *c += e;
            }
        }
        let n = embeddings.len() as f32;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for c in centroid.iter_mut() {
                *c /= norm;
            }
        }

        let now = chrono_now();
        Ok(Self {
            name: name.into(),
            centroid,
            num_segments: embeddings.len(),
            total_duration_secs: duration,
            created_at: now,
        })
    }

    /// Cosine similarity between this profile and an embedding.
    pub fn similarity(&self, embedding: &[f32]) -> f32 {
        cosine_similarity(&self.centroid, embedding)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile store — manages multiple profiles on disk
// ─────────────────────────────────────────────────────────────────────────────

/// Manages a directory of voice profiles. Each profile is a separate JSON file.
#[derive(Debug, Clone)]
pub struct ProfileStore {
    dir: PathBuf,
    profiles: Vec<VoiceProfile>,
}

impl ProfileStore {
    /// Load all profiles from the given directory.
    pub fn load(dir: &Path) -> Self {
        let mut profiles = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    match Self::load_profile(&path) {
                        Ok(p) => {
                            log::info!(
                                "Loaded profile '{}' ({} segments, {:.1}s)",
                                p.name,
                                p.num_segments,
                                p.total_duration_secs
                            );
                            profiles.push(p);
                        }
                        Err(e) => log::warn!("Failed to load {}: {}", path.display(), e),
                    }
                }
            }
        }

        log::info!(
            "Loaded {} voice profile(s) from {}",
            profiles.len(),
            dir.display()
        );
        Self {
            dir: dir.to_path_buf(),
            profiles,
        }
    }

    fn load_profile(path: &Path) -> Result<VoiceProfile> {
        let s = std::fs::read_to_string(path)
            .map_err(|_| ProfileError::NotFound(path.display().to_string()))?;
        let p: VoiceProfile =
            serde_json::from_str(&s).map_err(|e| ProfileError::ParseFailed(e.to_string()))?;
        Ok(p)
    }

    /// All loaded profiles.
    pub fn profiles(&self) -> &[VoiceProfile] {
        &self.profiles
    }

    /// Whether any profiles are enrolled.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Number of enrolled profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Add a new profile and save it to disk.
    pub fn add(&mut self, profile: VoiceProfile) -> Result<()> {
        let filename = format!("{}.json", sanitize_filename(&profile.name));
        let path = self.dir.join(&filename);
        self.save_profile(&profile, &path)?;
        self.profiles.push(profile);
        Ok(())
    }

    /// Rename a profile by index.
    pub fn rename(&mut self, index: usize, new_name: String) -> Result<()> {
        if index >= self.profiles.len() {
            anyhow::bail!("profile index out of bounds");
        }
        let old_name = self.profiles[index].name.clone();
        self.profiles[index].name = new_name.clone();
        self.save_all()?;
        log::info!("Renamed profile '{}' → '{}'", old_name, new_name);
        Ok(())
    }

    /// Delete a profile by index.
    pub fn delete(&mut self, index: usize) -> Result<()> {
        if index >= self.profiles.len() {
            anyhow::bail!("profile index out of bounds");
        }
        let removed = self.profiles.remove(index);
        // Remove all files and re-save (simple approach).
        self.save_all()?;
        log::info!("Deleted profile '{}'", removed.name);
        Ok(())
    }

    /// Find the best matching profile for an embedding.
    /// Returns `(max_similarity, is_any_match)` where is_any_match
    /// is true if any profile exceeds the threshold.
    pub fn best_similarity(&self, embedding: &[f32], threshold: f32) -> (f32, bool) {
        let mut max_sim = 0.0f32;
        for profile in &self.profiles {
            let sim = profile.similarity(embedding);
            max_sim = max_sim.max(sim);
        }
        (max_sim, max_sim >= threshold)
    }

    fn save_profile(&self, profile: &VoiceProfile, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create profile dir")?;
        }
        let json = serde_json::to_string_pretty(profile)?;
        std::fs::write(path, json).map_err(|e| ProfileError::SaveFailed(e.to_string()))?;
        log::info!("Saved profile '{}' to {}", profile.name, path.display());
        Ok(())
    }

    /// Re-save all profiles (after rename/delete).
    fn save_all(&self) -> Result<()> {
        // Clear directory.
        if self.dir.exists() {
            for entry in std::fs::read_dir(&self.dir)?.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "json").unwrap_or(false) {
                    let _ = std::fs::remove_file(path);
                }
            }
        }
        // Save each profile.
        for (i, profile) in self.profiles.iter().enumerate() {
            let filename = format!("{}_{}.json", i, sanitize_filename(&profile.name));
            let path = self.dir.join(&filename);
            self.save_profile(profile, &path)?;
        }
        Ok(())
    }
}

/// Generate a simple timestamp string without external crate.
fn chrono_now() -> String {
    // Use system time formatted as ISO-ish string.
    let now = std::time::SystemTime::now();
    let duration = now
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple date-time: not perfect but avoids adding chrono dependency.
    format!("{}", secs)
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
