# Audio I/O Fix + Device Selection + Test Recording

**Date:** 2026-03-22
**Status:** Draft

## Summary

Fix audio I/O to work with any device by capturing at the device's native sample rate and channels, converting to 16kHz mono at the I/O boundary. Add device selection to Settings UI. Add test recording to capture original and gated audio to WAV files.

## Motivation

- `capture.rs` hardcodes 16kHz mono — most Windows devices only support 48kHz stereo, causing "stream configuration not supported" errors
- No way to select input/output device — always uses system default
- No way to verify the gate is working correctly without external tools

## 1. Audio I/O Native Config

### Problem

`start_capture()` and `start_output()` hardcode `StreamConfig { channels: 1, sample_rate: 16000 }`. Devices reject this.

### Solution

Query the device's default config via `device.default_input_config()` and capture at native rate/channels. Convert at the I/O boundary:

**Input path:**
```
Mic (48kHz stereo) → cpal callback → stereo-to-mono → resample 48k→16k → pipeline (16kHz mono)
```

**Output path:**
```
Pipeline (16kHz mono) → resample 16k→48k → mono-to-stereo → cpal callback → Speakers (48kHz stereo)
```

The existing `src/audio/resampler.rs` (linear interpolation) handles resampling. Note: linear interpolation without anti-aliasing is lossy for downsampling, but acceptable for speech VAD/verification pipelines. Channel conversion: average all channels to mono on input, duplicate mono to all channels on output.

### Pipeline constants

The pipeline always runs at 16kHz mono, 512 samples/frame (32ms). These are constants, not configurable — the ML models expect this. The `AudioConfig.sample_rate`, `channels`, and `frame_samples` fields remain for pipeline configuration but are always 16000/1/512.

### Multi-channel handling

Devices may report 1, 2, or more channels. The conversion functions handle any channel count:
- **Input (N channels → mono):** average all N channels per sample
- **Output (mono → N channels):** duplicate the mono sample to all N channels

### Sample format

`device.default_input_config()` returns a `SupportedStreamConfig` with a `SampleFormat`. Use `build_input_stream` with explicit `f32` format — cpal handles conversion from the device's native format (I16, U16, etc.) automatically when using the typed callback.

### Changes

**`src/audio/capture.rs`:**
- `start_capture()` queries `device.default_input_config()` for native rate/channels
- New signature: `start_capture(device, tx) -> Result<(Stream, StreamConfig)>` — no sample_rate/frame_samples params
- Callback accumulates native-format samples, converts N-channel to mono, resamples native→16kHz, then drains the converted buffer in 512-sample pipeline frames
- Allocation note: the resample buffer is pre-allocated and reused across callbacks to avoid allocation on the real-time audio thread

**`src/audio/output.rs`:**
- `start_output()` queries `device.default_output_config()` for native rate/channels
- New signature: `start_output(device, rx) -> Result<Stream>` — no sample_rate param
- Feeder thread resamples 16kHz→native rate and expands mono→N channels *before* pushing into the shared deque. The deque contains samples in the device's native format. The cpal callback only does `pop_front()` (allocation-free).

**`src/audio/resampler.rs`:** Unchanged — already handles arbitrary rate conversion.

**`src/audio/mod.rs`:** Add `channels_to_mono(samples: &[f32], channels: u16) -> Vec<f32>` and `mono_to_channels(samples: &[f32], channels: u16) -> Vec<f32>` helper functions.

## 2. Device Selection

### Config Changes

Add optional device names to `AudioConfig`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,       // pipeline rate, always 16000
    pub channels: u16,          // pipeline channels, always 1
    pub frame_samples: usize,   // pipeline frame size, always 512
    #[serde(default)]
    pub input_device: Option<String>,   // None = system default
    #[serde(default)]
    pub output_device: Option<String>,  // None = system default
}
```

### Device Enumeration

New function in `src/audio/capture.rs`:
```rust
pub fn list_input_devices() -> Vec<String>
```

New function in `src/audio/output.rs`:
```rust
pub fn list_output_devices() -> Vec<String>
```

### Device Lookup

New functions:
```rust
pub fn find_input_device(name: &str) -> Result<Device>
pub fn find_output_device(name: &str) -> Result<Device>
```

Fall back to default if the named device is not found (log a warning).

### Settings UI

Add `ComboBox` dropdowns in `src/ui/settings_view.rs` under a new "Audio Devices" group:
- Input device: list from `list_input_devices()`, current selection from config
- Output device: list from `list_output_devices()`, current selection from config
- Device change takes effect on next Start (no hot-swap)

### App Startup

In `app.rs` `try_start()`:
- If `config.audio.input_device` is `Some(name)`, use `find_input_device(name)`
- Otherwise, use `default_input_device()`
- Same for output

## 3. Test Recording

### UI

Add a "Record Test" toggle button in the Dashboard (`main_view.rs`):
- Only visible when pipeline is running
- When toggled on: starts recording
- When toggled off or pipeline stops: finishes recording, closes files

`main_view::show()` gains two parameters:
- `is_recording: bool` — current recording state for button label
- `on_record_toggle: &mut dyn FnMut()` — callback when button is clicked

`app.rs` passes these from the shared `Arc<AtomicBool>` recording flag.

### Recording Logic

In `src/pipeline/processor.rs`:
- Add `recording: Option<TestRecorder>` field to `Processor`
- Add `recording_flag: Arc<AtomicBool>` field — shared with UI
- `process_frame()` checks the flag each frame:
  - false→true transition: create `TestRecorder`
  - true→false transition: call `recorder.finish()`
  - while true: write to both WAV files
- Use `Ordering::Relaxed` for the AtomicBool — exact frame boundaries don't matter

### TestRecorder

New file `src/pipeline/recorder.rs`:

```rust
pub struct TestRecorder {
    original: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
    gated: hound::WavWriter<std::io::BufWriter<std::fs::File>>,
}

impl TestRecorder {
    pub fn new() -> Result<Self>           // creates test_original.wav + test_gated.wav in CWD
    pub fn write_original(&mut self, frame: &[f32]) -> Result<()>
    pub fn write_gated(&mut self, frame: &[f32]) -> Result<()>
    pub fn finish(self) -> Result<()>      // flush + close with error reporting
}

impl Drop for TestRecorder {
    fn drop(&mut self) { /* safety net: finalize writers if finish() was not called */ }
}
```

WAV files are written to the current working directory: `test_original.wav` and `test_gated.wav`. Both are 16kHz mono f32.

### Communication

- `Arc<AtomicBool>` for `is_recording`, created in `app.rs`, shared with `Processor` and UI
- Processor owns the `TestRecorder` lifecycle — only the processor thread creates/finishes it
- When pipeline stops (`Processor` is dropped), `TestRecorder` is dropped too — the `Drop` impl ensures WAV files are finalized

## Files to Create

| File | Purpose |
|------|---------|
| `src/pipeline/recorder.rs` | `TestRecorder` — WAV recording of original + gated audio |

## Files to Modify

| File | Changes |
|------|---------|
| `src/audio/capture.rs` | Query native config, resample/convert in callback, device enumeration/lookup |
| `src/audio/output.rs` | Query native config, resample/convert in feeder, device enumeration/lookup |
| `src/audio/mod.rs` | Add `channels_to_mono()`, `mono_to_channels()` helpers |
| `src/config.rs` | Add `input_device`, `output_device` to `AudioConfig` |
| `src/ui/settings_view.rs` | Add device selection dropdowns |
| `src/ui/main_view.rs` | Add "Record Test" toggle button |
| `src/app.rs` | Device lookup in `try_start()`, recording flag |
| `src/pipeline/processor.rs` | Recording integration, `TestRecorder` lifecycle |
| `src/pipeline/mod.rs` | Add `mod recorder` |

## Files Unchanged

| File | Reason |
|------|--------|
| `src/audio/resampler.rs` | Already handles arbitrary rate conversion |
| `src/inference.rs` | No audio dependency |
| `src/vad/*` | No audio dependency |
| `src/speaker/*` | No audio dependency |
| `src/pipeline/state_machine.rs` | Pure logic |

## Error Handling

- Device not found by name → warn + fall back to default
- Native config query fails → return error with device name
- WAV write fails → log warning, disable recording, don't crash pipeline
- Resampling edge cases (same rate) → pass through unchanged (already handled)
