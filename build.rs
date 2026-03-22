//! Build script — converts ONNX models into native Burn Rust code.
//!
//! # Strict mode (default)
//!
//! Missing or broken ONNX models produce a **compile error** with a clear
//! diagnostic message showing what's wrong and how to fix it.
//!
//! # Dev mode (`--features dev`)
//!
//! Errors are downgraded to warnings. Energy-based VAD fallback is used,
//! and speaker verification is disabled. Useful for UI/audio iteration.

use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    println!("cargo:rerun-if-changed=models/");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let is_dev = env::var("CARGO_FEATURE_DEV").is_ok();

    let mut diag = Diagnostics::new(is_dev);

    // ── Validate models/ directory ──────────────────────────────────
    let models_dir = Path::new("models");
    if !models_dir.exists() {
        diag.error(
            "MISSING_MODELS_DIR",
            "Directory `models/` does not exist.",
            &["Create it: mkdir models", "Then download ONNX files (see README.md)"],
        );
    }

    // ── Silero VAD ──────────────────────────────────────────────────
    compile_model(
        &mut diag,
        Path::new("models/silero_vad.onnx"),
        "silero_vad",
        "model/",
        1024,        // min file size (bytes)
        "has_silero_vad",
        &[
            "Download it:",
            "  wget -O models/silero_vad.onnx \\",
            "    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        ],
        &[
            "This usually means the model uses ONNX operators not yet in burn-onnx.",
            "Possible fixes:",
            "  1. Try Silero VAD v4 (simpler graph, fewer ops)",
            "  2. Build with --features dev to use energy-based fallback",
            "  3. Report the missing operator: https://github.com/tracel-ai/burn-onnx/issues",
        ],
        &[
            "File is suspiciously small — likely a Git LFS pointer or corrupt download.",
            "Re-download:",
            "  wget -O models/silero_vad.onnx \\",
            "    https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        ],
    );

    // ── ECAPA-TDNN ──────────────────────────────────────────────────
    compile_model(
        &mut diag,
        Path::new("models/ecapa_tdnn.onnx"),
        "ecapa_tdnn",
        "model/",
        10_000,      // min file size
        "has_ecapa_tdnn",
        &[
            "Option 1 — WeSpeaker (recommended, has ready ONNX):",
            "  git lfs install",
            "  git clone https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM /tmp/ecapa",
            "  cp /tmp/ecapa/*.onnx models/ecapa_tdnn.onnx",
            "",
            "Option 2 — Export from SpeechBrain (requires Python):",
            "  See README.md for the export script.",
        ],
        &[
            "ECAPA-TDNN may use operators not yet in burn-onnx.",
            "Possible fixes:",
            "  1. Try WeSpeaker export (often simpler graph):",
            "     https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM",
            "  2. Export from SpeechBrain with opset 14 (fewer dynamic ops)",
            "  3. Build with --features dev to disable speaker verification",
        ],
        &[
            "File is suspiciously small — likely a Git LFS pointer.",
            "If you cloned from HuggingFace, run: git lfs install && git lfs pull",
        ],
    );

    // ── Emit all diagnostics ────────────────────────────────────────
    diag.emit(&out_dir);
}

// ─────────────────────────────────────────────────────────────────────────────
// Model compilation
// ─────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn compile_model(
    diag: &mut Diagnostics,
    onnx_path: &Path,
    model_name: &str,
    out_subdir: &str,
    min_file_size: u64,
    cfg_flag: &str,
    missing_hints: &[&str],
    conversion_hints: &[&str],
    corrupt_hints: &[&str],
) {
    let display_name = model_name.to_uppercase().replace('_', "-");

    if !onnx_path.exists() {
        diag.error(
            &format!("{}_MISSING", model_name.to_uppercase()),
            &format!("{} not found.", onnx_path.display()),
            missing_hints,
        );
        return;
    }

    let file_size = fs::metadata(onnx_path).map(|m| m.len()).unwrap_or(0);

    if file_size < min_file_size {
        diag.error(
            &format!("{}_CORRUPT", model_name.to_uppercase()),
            &format!(
                "{} is only {} bytes (expected ≥ {} bytes).",
                onnx_path.display(), file_size, min_file_size,
            ),
            corrupt_hints,
        );
        return;
    }

    // Attempt compilation with panic capture for clear error reporting.
    match try_compile_onnx(onnx_path, out_subdir) {
        Ok(()) => {
            println!("cargo:rustc-cfg={}", cfg_flag);
            let size_str = if file_size > 1_000_000 {
                format!("{:.1} MB", file_size as f64 / (1024.0 * 1024.0))
            } else {
                format!("{:.1} KB", file_size as f64 / 1024.0)
            };
            diag.success(
                &format!("{}_OK", model_name.to_uppercase()),
                &format!("{} compiled ({} → native Rust)", display_name, size_str),
            );
        }
        Err(msg) => {
            diag.error(
                &format!("{}_CONVERSION_FAILED", model_name.to_uppercase()),
                &format!(
                    "burn-onnx failed to convert {}:\n\n  {}",
                    onnx_path.display(),
                    msg.replace('\n', "\n  "),
                ),
                conversion_hints,
            );
        }
    }
}

/// Run burn-onnx ModelGen inside catch_unwind to capture panics.
fn try_compile_onnx(onnx_path: &Path, out_subdir: &str) -> Result<(), String> {
    let path_str = onnx_path
        .to_str()
        .ok_or("non-UTF-8 model path")?
        .to_string();
    let out = out_subdir.to_string();

    std::panic::catch_unwind(move || {
        burn_import::onnx::ModelGen::new()
            .input(&path_str)
            .out_dir(&out)
            .run_from_script();
    })
    .map_err(|payload| {
        if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else if let Some(s) = payload.downcast_ref::<&str>() {
            s.to_string()
        } else {
            "unknown panic (no message available)".to_string()
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics collector
// ─────────────────────────────────────────────────────────────────────────────

struct DiagnosticEntry {
    code: String,
    level: DiagLevel,
    message: String,
    hints: Vec<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum DiagLevel { Success, Warning, Error }

struct Diagnostics {
    entries: Vec<DiagnosticEntry>,
    is_dev: bool,
}

impl Diagnostics {
    fn new(is_dev: bool) -> Self {
        Self { entries: Vec::new(), is_dev }
    }

    fn success(&mut self, code: &str, message: &str) {
        self.entries.push(DiagnosticEntry {
            code: code.to_string(),
            level: DiagLevel::Success,
            message: message.to_string(),
            hints: Vec::new(),
        });
    }

    fn error(&mut self, code: &str, message: &str, hints: &[&str]) {
        let level = if self.is_dev { DiagLevel::Warning } else { DiagLevel::Error };
        self.entries.push(DiagnosticEntry {
            code: code.to_string(), level,
            message: message.to_string(),
            hints: hints.iter().map(|s| s.to_string()).collect(),
        });
    }

    /// Write all diagnostics to cargo output + generate `diagnostics.rs`.
    fn emit(self, out_dir: &Path) {
        let mut diag_rs = String::from("// Auto-generated build diagnostics. Do not edit.\n\n");
        let mut has_errors = false;

        // ── Pretty-printed cargo warnings ───────────────────────────
        println!("cargo:warning=");
        println!("cargo:warning=┌──────────────────────────────────────────────┐");
        println!("cargo:warning=│        Voice Gate — Build Diagnostics        │");
        println!("cargo:warning=└──────────────────────────────────────────────┘");
        println!("cargo:warning=");

        if self.is_dev {
            println!("cargo:warning=  Mode: DEV (errors downgraded to warnings)");
            println!("cargo:warning=");
        }

        for entry in &self.entries {
            let icon = match entry.level {
                DiagLevel::Success => "OK ",
                DiagLevel::Warning => "WRN",
                DiagLevel::Error   => "ERR",
            };

            println!("cargo:warning=  [{}] [{}] {}", icon, entry.code, entry.message);
            for hint in &entry.hints {
                if hint.is_empty() {
                    println!("cargo:warning=");
                } else {
                    println!("cargo:warning=         {}", hint);
                }
            }
            println!("cargo:warning=");

            if entry.level == DiagLevel::Error {
                has_errors = true;

                // Build compile_error! message.
                let mut msg = format!(
                    "\n\n========================================\n\
                     voice-gate build error: {}\n\
                     ========================================\n\n\
                     {}\n",
                    entry.code, entry.message,
                );
                if !entry.hints.is_empty() {
                    msg.push_str("\nHow to fix:\n");
                    for h in &entry.hints {
                        if h.is_empty() { msg.push('\n'); }
                        else { writeln!(&mut msg, "  {}", h).unwrap(); }
                    }
                }
                msg.push_str("\nTo skip this check: cargo build --features dev\n\n");

                let escaped = msg
                    .replace('\\', "\\\\")
                    .replace('"', "\\\"")
                    .replace('\n', "\\n");
                writeln!(&mut diag_rs, "compile_error!(\"{}\");", escaped).unwrap();
            }
        }

        if !has_errors {
            writeln!(&mut diag_rs, "// All models compiled successfully. No issues found.").unwrap();
        }

        // ── Write to $OUT_DIR/model/diagnostics.rs ──────────────────
        let model_dir = out_dir.join("model");
        let _ = fs::create_dir_all(&model_dir);
        let diag_path = model_dir.join("diagnostics.rs");
        fs::write(&diag_path, diag_rs).expect("failed to write diagnostics.rs");
    }
}
