# Voice Gate (Burn Edition)

**Intelligent microphone gate that activates only for *your* voice.**

Built entirely in Rust with the [Burn](https://burn.dev) deep learning framework —
**no ONNX Runtime, no C++ bindings, no shared libraries**. ONNX models are
compiled into native Rust code at build time and run on any Burn backend.

---

## How It Works

```
┌─────────────┐      ┌───────────┐      ┌──────────────┐      ┌────────────┐
│  Microphone  │─────▶│  Silero   │─────▶│  ECAPA-TDNN  │─────▶│   Output   │
│   (cpal)     │ PCM  │   VAD     │ voice│  Speaker ID  │ mine │  (cpal /   │
│              │      │  (Burn)   │      │   (Burn)     │      │   WAV)     │
└─────────────┘      └─────┬─────┘      └──────┬───────┘      └────────────┘
                       no voice             not mine
                           │                    │
                           ▼                    ▼
                        [silence]           [silence]
```

## Why Burn?

| Feature           | ONNX Runtime (ort crate)           | Burn                                |
|-------------------|------------------------------------|-------------------------------------|
| C++ dependency    | Yes (libonnxruntime.so/.dll)       | **None — pure Rust**                |
| Build complexity  | Must ship/find shared library      | `cargo build` and done              |
| GPU support       | CUDA only                          | **CUDA + Vulkan + Metal + DX12**    |
| WASM support      | No                                 | Yes (NdArray or WGPU backends)      |
| Model format      | .onnx loaded at runtime            | Compiled to Rust at build time      |
| Cross-compilation | Painful                            | Standard `cargo` cross-compile      |

## Architecture

```
src/
├── main.rs               # Entry point
├── app.rs                # egui App — wires audio + pipeline + UI
├── backend.rs            # Burn backend selection (CPU/WGPU/CUDA)
├── config.rs             # Tuneable parameters (persisted to JSON)
├── error.rs              # Error types
│
├── model/                # Auto-generated Burn code (from build.rs)
│   └── mod.rs            # include!() of generated silero_vad.rs / ecapa_tdnn.rs
│
├── audio/
│   ├── capture.rs        # Microphone input (cpal)
│   ├── output.rs         # Audio output (cpal)
│   └── resampler.rs      # Sample-rate conversion
│
├── vad/
│   └── silero.rs         # Silero VAD wrapper (Burn tensor ops)
│
├── speaker/
│   ├── embedding.rs      # ECAPA-TDNN wrapper (Burn tensor ops)
│   ├── enrollment.rs     # Voice enrollment workflow
│   └── profile.rs        # Voice profile persistence
│
├── pipeline/
│   ├── processor.rs      # VAD → Speaker Verification → Gate
│   └── state_machine.rs  # Silent / MyVoice / OtherVoice / Trailing
│
└── ui/
    ├── main_view.rs      # Dashboard with level meters
    ├── enrollment_view.rs# Enrollment wizard
    └── settings_view.rs  # Threshold sliders
```

## Prerequisites

| Requirement      | Version | Notes                                  |
|------------------|---------|----------------------------------------|
| Rust toolchain   | ≥ 1.80  | `rustup update stable`                 |
| ALSA dev headers | —       | Linux only: `apt install libasound2-dev` |
| CUDA toolkit     | ≥ 12.0  | Only for `--features cuda`             |

## Setup

### 1. Download ONNX models into `models/`

```bash
# Silero VAD (direct download from GitHub)
wget -O models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

# ECAPA-TDNN (WeSpeaker variant, 512-d embeddings)
git lfs install
git clone https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM /tmp/ecapa
cp /tmp/ecapa/*.onnx models/ecapa_tdnn.onnx
```

### 2. Build & Run

```bash
# CPU (pure Rust, no GPU needed)
cargo run --release

# GPU via Vulkan/Metal/DX12 (works on AMD, Intel, NVIDIA)
cargo run --release --features wgpu

# NVIDIA CUDA
cargo run --release --features cuda
```

The **first build** takes longer because `burn-onnx` compiles the ONNX models
into Rust source code. Subsequent builds are fast (incremental).

### 3. Enroll Your Voice

1. Go to the **Enrollment** tab in the GUI.
2. Click **Start Recording** and speak for ≥ 10 seconds.
3. Click **Finish & Save** — your voice profile is stored in `profiles/default.json`.
4. Return to **Dashboard** and click **Start**.

Now only your voice passes through. Other voices, noise, and background are silenced.

## Tuning

| Parameter              | Default | Effect                                   |
|------------------------|---------|------------------------------------------|
| `vad.threshold`        | 0.5     | Silero speech confidence cutoff           |
| `speaker.similarity`   | 0.70    | Cosine similarity to accept as "you"      |
| `gate.hold_time_ms`    | 300     | Keep gate open after speech ends          |
| `gate.pre_buffer_ms`   | 100     | Audio kept before speech onset            |

All adjustable live from the **Settings** tab.

## What If ONNX Models Are Missing?

The project builds and runs without models:

- **No `silero_vad.onnx`** → Falls back to RMS energy-based detection (less accurate).
- **No `ecapa_tdnn.onnx`** → Speaker verification disabled (all speech passes through).

This lets you iterate on UI and audio code without needing models.

## License

MIT
