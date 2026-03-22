//! Model discovery and download.
//!
//! Checks whether required ONNX models exist on disk and downloads them
//! from their canonical URLs if missing.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

/// Silero VAD v5.1.2 — voice activity detection.
const SILERO_VAD_URL: &str =
    "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx";

/// WeSpeaker ECAPA-TDNN 512 — speaker embedding (192-dim).
const ECAPA_TDNN_URL: &str =
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-ecapa-tdnn512/resolve/main/voxceleb_ECAPA512.onnx";

/// Expected model file name for Silero VAD.
const SILERO_VAD_FILENAME: &str = "silero_vad.onnx";

/// Expected model file name for ECAPA-TDNN.
const ECAPA_TDNN_FILENAME: &str = "ecapa_tdnn.onnx";

/// Required model definition.
struct ModelDef {
    filename: &'static str,
    url: &'static str,
    description: &'static str,
}

/// All models required by the application.
const REQUIRED_MODELS: &[ModelDef] = &[
    ModelDef {
        filename: SILERO_VAD_FILENAME,
        url: SILERO_VAD_URL,
        description: "Silero VAD v5 (voice activity detection)",
    },
    ModelDef {
        filename: ECAPA_TDNN_FILENAME,
        url: ECAPA_TDNN_URL,
        description: "ECAPA-TDNN 512 (speaker embedding)",
    },
];

/// Status of model readiness.
#[derive(Debug, Clone)]
pub enum ModelStatus {
    /// All models present and ready.
    Ready,
    /// Some models are missing — user action required.
    Missing(Vec<MissingModel>),
    /// Models are being downloaded.
    Downloading {
        current_model: String,
        progress: f32,
    },
    /// Download completed successfully.
    DownloadComplete,
    /// Download or validation failed.
    Error(String),
}

/// A model that was not found on disk.
#[derive(Debug, Clone)]
pub struct MissingModel {
    pub filename: String,
    pub description: String,
}

/// Shared download progress, updated from the download thread.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub status: ModelStatus,
}

/// Checks which models are present in `models_dir`.
pub fn check_models(models_dir: &Path) -> ModelStatus {
    let missing: Vec<MissingModel> = REQUIRED_MODELS
        .iter()
        .filter(|m| !models_dir.join(m.filename).exists())
        .map(|m| MissingModel {
            filename: m.filename.to_string(),
            description: m.description.to_string(),
        })
        .collect();

    if missing.is_empty() {
        ModelStatus::Ready
    } else {
        ModelStatus::Missing(missing)
    }
}

/// Returns the full path for the Silero VAD model.
pub fn silero_vad_path(models_dir: &Path) -> PathBuf {
    models_dir.join(SILERO_VAD_FILENAME)
}

/// Returns the full path for the ECAPA-TDNN model.
pub fn ecapa_tdnn_path(models_dir: &Path) -> PathBuf {
    models_dir.join(ECAPA_TDNN_FILENAME)
}

/// Download all missing models into `models_dir`.
///
/// Updates `progress` after each chunk so the UI can display a progress bar.
/// Runs synchronously — call from a background thread.
pub fn download_models(models_dir: &Path, progress: Arc<Mutex<DownloadProgress>>) -> Result<()> {
    std::fs::create_dir_all(models_dir).with_context(|| {
        format!(
            "failed to create models directory: {}",
            models_dir.display()
        )
    })?;

    for model in REQUIRED_MODELS {
        let dest = models_dir.join(model.filename);
        if dest.exists() {
            continue;
        }

        {
            let mut p = progress.lock().unwrap();
            p.status = ModelStatus::Downloading {
                current_model: model.description.to_string(),
                progress: 0.0,
            };
        }

        download_file(model.url, &dest, model.description, &progress)
            .with_context(|| format!("failed to download {}", model.description))?;
    }

    {
        let mut p = progress.lock().unwrap();
        p.status = ModelStatus::DownloadComplete;
    }
    Ok(())
}

/// Download a single file with progress tracking.
fn download_file(
    url: &str,
    dest: &Path,
    description: &str,
    progress: &Arc<Mutex<DownloadProgress>>,
) -> Result<()> {
    log::info!("Downloading {} from {}", description, url);

    let response = ureq::get(url)
        .call()
        .with_context(|| format!("HTTP request failed for {}", url))?;

    let total_size: Option<u64> = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());

    let tmp_dest = dest.with_extension("onnx.tmp");
    let mut file = std::fs::File::create(&tmp_dest)
        .with_context(|| format!("failed to create {}", tmp_dest.display()))?;

    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 65536];
    let mut body = response.into_body();
    let mut reader = body.as_reader();

    loop {
        let n = reader
            .read(&mut buf)
            .context("error reading download stream")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;

        if let Some(total) = total_size {
            let frac = downloaded as f32 / total as f32;
            let mut p = progress.lock().unwrap();
            p.status = ModelStatus::Downloading {
                current_model: description.to_string(),
                progress: frac,
            };
        }
    }
    file.flush()?;
    drop(file);

    // Atomic rename: tmp → final.
    std::fs::rename(&tmp_dest, dest).with_context(|| {
        format!(
            "failed to rename {} → {}",
            tmp_dest.display(),
            dest.display()
        )
    })?;

    log::info!("Downloaded {} ({} bytes)", description, downloaded);
    Ok(())
}
