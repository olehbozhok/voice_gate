//! Voice profile — stored speaker embedding centroid.

use std::path::Path;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use super::cosine_similarity;
use crate::error::ProfileError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub name: String,
    pub centroid: Vec<f32>,
    pub num_segments: usize,
    pub total_duration_secs: f32,
}

impl VoiceProfile {
    /// Average multiple embeddings into a single centroid.
    pub fn from_embeddings(name: impl Into<String>, embeddings: &[Vec<f32>], duration: f32) -> Result<Self> {
        if embeddings.is_empty() { anyhow::bail!("zero embeddings"); }
        let dim = embeddings[0].len();
        let mut centroid = vec![0.0f32; dim];
        for emb in embeddings {
            for (c, e) in centroid.iter_mut().zip(emb.iter()) { *c += e; }
        }
        let n = embeddings.len() as f32;
        for c in centroid.iter_mut() { *c /= n; }
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 { for c in centroid.iter_mut() { *c /= norm; } }
        Ok(Self { name: name.into(), centroid, num_segments: embeddings.len(), total_duration_secs: duration })
    }

    pub fn similarity(&self, embedding: &[f32]) -> f32 {
        cosine_similarity(&self.centroid, embedding)
    }

    pub fn is_match(&self, embedding: &[f32], threshold: f32) -> bool {
        self.similarity(embedding) >= threshold
    }

    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)
            .map_err(|_| ProfileError::NotFound(path.display().to_string()))?;
        let p: Self = serde_json::from_str(&s)
            .map_err(|e| ProfileError::ParseFailed(e.to_string()))?;
        log::info!("Loaded profile '{}' ({} segments, {:.1}s)", p.name, p.num_segments, p.total_duration_secs);
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create profile dir")?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json).map_err(|e| ProfileError::SaveFailed(e.to_string()))?;
        log::info!("Saved profile '{}' to {}", self.name, path.display());
        Ok(())
    }
}
