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

## Project Structure

- `src/inference.rs` — ONNX runtime abstraction. Only file that imports tract. To swap runtimes, change only this file.
- `src/vad/` — voice activity detection.
- `src/speaker/` — speaker embedding, enrollment, profiles.
- `src/pipeline/` — audio processing pipeline and gate state machine.
- `src/audio/` — audio I/O via cpal.
- `src/ui/` — egui views.
