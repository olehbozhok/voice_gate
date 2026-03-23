# CLAUDE.md

## Code Quality Standards

Write code as a highly qualified senior engineer with years of experience. Every piece of code should be:

- **Readable** — code is read far more often than written. Prefer clarity over cleverness.
- **Well-structured** — each module, function, and type has a single clear responsibility.
- **Right level of abstraction** — not too low (boilerplate), not too high (over-engineering). Abstract when there's a real reason, inline when it's simpler.
- **Self-documenting** — good names replace comments. Add comments only when the *why* isn't obvious from the code.
- **Minimal** — no dead code, no speculative features, no premature generalization. YAGNI.

## Rust Conventions

- Use `anyhow::Result` for application errors, `thiserror` for library-style error enums.
- Prefer concrete types over trait objects unless polymorphism is needed.
- Keep functions short — if a function needs a comment block explaining its sections, split it.
- Use `log` crate macros for diagnostics (`info!`, `warn!`, `error!`, `trace!`).
- **No magic numbers.** All constants must be declared as `const` with a doc comment explaining the value. Example: `/// Pipeline sample rate expected by ML models (Silero VAD, ECAPA-TDNN). const PIPELINE_SAMPLE_RATE: u32 = 16_000;`

## Project Structure

- `src/inference.rs` — ONNX runtime abstraction. Only file that imports `ort`. To swap runtimes, change only this file.
- `src/vad/silero.rs` — Silero VAD v5: 512-sample frames + 64-sample context, LSTM state carry.
- `src/speaker/embedding.rs` — ECAPA-TDNN: raw audio → mel-spectrogram (`mel.rs`) → 192-dim embedding.
- `src/speaker/profile.rs` — `VoiceProfile` (centroid + metadata), `ProfileStore` (multiple profiles, JSON in `%APPDATA%/voice-gate/profiles/`).
- `src/speaker/enrollment.rs` — `EnrollmentSession`: accumulates voiced segments, extracts embeddings from sliding windows, averages into centroid.
- `src/pipeline/processor.rs` — main audio loop. VAD every frame, feeds verifier, evaluates `GateMode`, manages pre-buffer and enrollment.
- `src/pipeline/verifier.rs` — `SpeakerVerifier`: background thread, computes embeddings, compares against `Arc<RwLock<ProfileStore>>` (live updates).
- `src/pipeline/recorder.rs` — `TestRecorder`: writes original + gated WAV files.
- `src/config.rs` — all config structs + `GateMode` enum with `evaluate()`. Shared via `Arc<RwLock<Config>>`.
- `src/audio/capture.rs` — cpal input: captures at device native rate, sends both original and 16kHz mono downsampled.
- `src/audio/output.rs` — cpal output: plays native audio directly (no resampling).
- `src/audio/resampler.rs` — linear interpolation resampler.
- `src/app.rs` — `VoiceGateApp`: eframe app, owns shared state, background model loading, pipeline lifecycle.
- `src/ui/main_view.rs` — Dashboard: gate status, start/stop, input level, details with colored telemetry.
- `src/ui/settings_view.rs` — Settings: thresholds, gate mode, hold time, pre-buffer, device selection.
- `src/ui/enrollment_view.rs` — Enrollment: record voice, manage multiple profiles (rename, delete).

## Architecture Notes

- **Audio path**: capture sends original quality + 16kHz downsampled. Pipeline runs ML on 16kHz. Gate passes original audio to output — no double-resampling.
- **GateMode**: `Optimistic` (open first, verify later), `Strict` (verify first), `VadOnly` (no speaker verification). Each mode's `evaluate()` is a pure function — all state in `GateInput`, trivial to test.
- **Shared state**: `Config`, `ProfileStore`, `PipelineTelemetry` via `Arc<RwLock<T>>`. Settings apply instantly, no restart needed.
- **Threading**: UI thread (egui), processor thread (VAD + gate), verifier thread (ECAPA-TDNN), model loading thread.
- **Models**: downloaded on first launch to `%APPDATA%/voice-gate/models/`. Silero VAD (~2.3MB), ECAPA-TDNN (~24.9MB).
