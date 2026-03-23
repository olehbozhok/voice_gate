# Voice Gate

**Microphone gate that only passes *your* voice through.**

Voice Gate sits between your microphone and your headphones (or any audio output). It uses two neural networks to decide in real time whether to let audio through: one detects speech, the other identifies the speaker. If it's you — audio passes. If it's anyone else, background noise, or silence — it's muted.

Useful for:
- **Calls and meetings** — mutes your mic when someone next to you talks, your TV plays, or the dog barks.
- **Streaming** — keeps only your voice on the stream without a physical mute button.
- **Recording** — clean voice capture without ambient contamination.

---

## How it works

```
Microphone (48kHz native)
    │
    ├── original audio ──────────────────────────────────┐
    │                                                    │
    └── downsampled 16kHz mono                           │
         │                                               │
         ├── Silero VAD ──── speech? ─────┐              │
         │                                │              │
         └── ECAPA-TDNN ── is it you? ────┤              │
                                          │              │
                                     GateMode ───── pass/block
                                          │              │
                                          │              ▼
                                          └─────── Headphones (48kHz)
```

Two key details:
1. **Original audio quality is preserved.** The 16kHz copy is only for the neural networks. What you hear (or what goes to output) is the untouched 48kHz signal — no double-resampling artifacts.
2. **The gate decision is a pure function.** All state goes in, a pass/block decision comes out. Easy to test, easy to reason about.

## Gate modes

| Mode | Behaviour | Trade-off |
|------|-----------|-----------|
| **Optimistic** (default) | Opens instantly on speech, closes if verification says "not you" | Your voice is never clipped. Other voices may leak for ~1s. |
| **Strict** | Stays closed until verification confirms you | Other voices never leak. Your first ~1s may be lost (mitigated by pre-buffer). |
| **VAD Only** | Passes all speech, blocks silence | No speaker verification. Useful without an enrolled profile. |

## Setup

### Build and run

```bash
cargo build --release
cargo run --release
```

On first launch, Voice Gate downloads the required ONNX models (~27 MB total) to `%APPDATA%/voice-gate/models/`. No manual setup needed.

### Enroll your voice

1. Go to **Dashboard** → click **Start** to begin the audio pipeline.
2. Switch to the **Enrollment** tab.
3. Click **Start Recording** and speak naturally for 10+ seconds.
4. Click **Finish & Save**.

Your voice profile is stored in `%APPDATA%/voice-gate/profiles/`. You can create multiple profiles and rename or delete them from the Enrollment tab.

## Settings

All parameters are adjustable live — no restart required (except audio device changes).

| Parameter | Default | What it does |
|-----------|---------|-------------|
| Speech threshold | 0.50 | VAD confidence cutoff. Higher = fewer false triggers. |
| Similarity threshold | 0.70 | How closely the voice must match your profile. Higher = stricter. |
| Hold time | 300 ms | Keeps gate open after speech ends (prevents clipping word tails). |
| Pre-buffer | 100 ms | Delay line that captures word onsets before the gate opens. |
| Verification settle | 500 ms | Grace period before trusting verification in Optimistic mode. |

## Models

| Model | Size | Purpose |
|-------|------|---------|
| [Silero VAD v5](https://github.com/snakers4/silero-vad) | 2.3 MB | Voice activity detection (512-sample frames at 16kHz) |
| [ECAPA-TDNN](https://github.com/wenet-e2e/wespeaker) (WeSpeaker) | 24.9 MB | Speaker embedding extraction (192-dim, expects mel features) |

Downloaded automatically on first launch.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Rust ≥ 1.80 | `rustup update stable` |
| ALSA dev headers | Linux only: `apt install libasound2-dev` |

## Using with Discord / Zoom / OBS

Voice Gate outputs audio to your headphones for monitoring. To use it as a virtual microphone in other apps, pair it with **VB-CABLE** — a free virtual audio device.

### Setup

1. Download VB-CABLE from [vb-audio.com/Cable](https://vb-audio.com/Cable/) (free, donationware).
2. Unzip and run `VBCABLE_Setup_x64.exe` as administrator. Reboot.
3. Two new audio devices appear: **CABLE Input** (output) and **CABLE Output** (input).

### Configuration

| App | Setting |
|-----|---------|
| **Voice Gate** (Settings tab) | Output device → **CABLE Input** |
| **Discord / Zoom / OBS** | Input device → **CABLE Output** |

Voice Gate writes gated audio to CABLE Input. The virtual cable forwards it to CABLE Output, which other apps see as a microphone. Only your voice gets through.

## About

Written with the help of AI (Claude), but with deliberate architectural decisions and hands-on testing — not generated slop. Every gate mode, every buffer strategy, every threading boundary was discussed, tested with real audio, and iterated on based on actual behaviour.

## License

MIT
