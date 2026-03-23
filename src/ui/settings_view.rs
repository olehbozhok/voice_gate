//! Settings — threshold sliders, device selection, backend info.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{RichText, Ui};
use parking_lot::RwLock;

use crate::config::{Config, GateMode, OptimisticConfig};

/// Minimum interval between device list refreshes.
const DEVICE_REFRESH_INTERVAL: Duration = Duration::from_secs(3);

/// Cached audio device lists, refreshed in a background thread.
pub struct DeviceListCache {
    inner: Arc<RwLock<DeviceListInner>>,
}

struct DeviceListInner {
    input_devices: Vec<String>,
    output_devices: Vec<String>,
    last_refresh: Instant,
    refreshing: bool,
}

impl DeviceListCache {
    /// Create with an initial (blocking) refresh.
    pub fn new() -> Self {
        let inner = DeviceListInner {
            input_devices: crate::audio::capture::list_input_devices(),
            output_devices: crate::audio::output::list_output_devices(),
            last_refresh: Instant::now(),
            refreshing: false,
        };
        Self {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    /// Trigger a background refresh if enough time has passed.
    /// Calls `ctx.request_repaint()` when the refresh completes.
    pub fn request_refresh(&self, ctx: &egui::Context) {
        let should_refresh = {
            let inner = self.inner.read();
            !inner.refreshing && inner.last_refresh.elapsed() >= DEVICE_REFRESH_INTERVAL
        };

        if !should_refresh {
            return;
        }

        {
            self.inner.write().refreshing = true;
        }

        let inner = self.inner.clone();
        let ctx = ctx.clone();
        std::thread::Builder::new()
            .name("device-refresh".into())
            .spawn(move || {
                let input = crate::audio::capture::list_input_devices();
                let output = crate::audio::output::list_output_devices();
                {
                    let mut guard = inner.write();
                    guard.input_devices = input;
                    guard.output_devices = output;
                    guard.last_refresh = Instant::now();
                    guard.refreshing = false;
                }
                ctx.request_repaint();
            })
            .ok();
    }

    /// Read the cached device lists (non-blocking).
    pub fn input_devices(&self) -> Vec<String> {
        self.inner.read().input_devices.clone()
    }

    /// Read the cached device lists (non-blocking).
    pub fn output_devices(&self) -> Vec<String> {
        self.inner.read().output_devices.clone()
    }
}

/// Result of showing settings UI.
pub struct SettingsResult {
    /// Any setting was modified (triggers config save).
    pub changed: bool,
    /// Audio device was changed (triggers pipeline restart).
    pub device_changed: bool,
}

/// Persistent UI state for settings that must survive mode switches.
pub struct SettingsViewState {
    /// Preserved OptimisticConfig so switching away and back doesn't lose values.
    pub optimistic_cfg: OptimisticConfig,
}

impl SettingsViewState {
    pub fn from_config(config: &Config) -> Self {
        let optimistic_cfg = match config.gate.mode {
            GateMode::Optimistic(cfg) => cfg,
            _ => OptimisticConfig::default(),
        };
        Self { optimistic_cfg }
    }
}

/// Returns what changed in settings.
pub fn show(
    ui: &mut Ui,
    config: &mut Config,
    devices: &DeviceListCache,
    ctx: &egui::Context,
    state: &mut SettingsViewState,
) -> SettingsResult {
    let mut changed = false;
    let mut device_changed = false;

    // Sync: if currently Optimistic, keep state in sync with config.
    if let GateMode::Optimistic(cfg) = config.gate.mode {
        state.optimistic_cfg = cfg;
    }
    ui.heading("Settings");
    ui.add_space(8.0);

    ui.group(|ui| {
        ui.label(RichText::new("Voice Activity Detection").strong());
        ui.horizontal(|ui| {
            ui.label("Speech threshold:");
            if ui
                .add(egui::Slider::new(&mut config.vad.threshold, 0.1..=0.95).step_by(0.05))
                .changed()
            {
                changed = true;
            }
        });
        ui.label(
            egui::RichText::new("Higher = fewer false positives, may miss quiet speech.")
                .small()
                .weak(),
        );
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Speaker Verification").strong());
        ui.horizontal(|ui| {
            ui.label("Similarity threshold:");
            if ui
                .add(
                    egui::Slider::new(&mut config.speaker.similarity_threshold, 0.40..=0.95)
                        .step_by(0.05),
                )
                .changed()
            {
                changed = true;
            }
        });
        ui.label(
            egui::RichText::new(
                "Lower = permissive. Higher = strict (may reject unusual intonation).",
            )
            .small()
            .weak(),
        );
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Gate Behaviour").strong());
        ui.horizontal(|ui| {
            ui.label("Hold time (ms):");
            let mut hold = config.gate.hold_time_ms as f32;
            if ui.add(egui::Slider::new(&mut hold, 50.0..=1000.0).step_by(50.0)).changed() {
                config.gate.hold_time_ms = hold as u32; changed = true;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Pre-buffer (ms):").on_hover_text("TODO: not yet implemented. Will add latency to avoid clipping word starts in Strict mode.");
            let mut pre = config.gate.pre_buffer_ms as f32;
            if ui.add(egui::Slider::new(&mut pre, 0.0..=500.0).step_by(25.0)).changed() {
                config.gate.pre_buffer_ms = pre as u32; changed = true;
            }
        });
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Verification mode:");
            let is_optimistic = matches!(config.gate.mode, GateMode::Optimistic(_));
            if ui.selectable_label(is_optimistic, "Optimistic").clicked() && !is_optimistic {
                config.gate.mode = GateMode::Optimistic(state.optimistic_cfg);
                changed = true;
            }
            if ui.selectable_label(config.gate.mode == GateMode::Strict, "Strict").clicked() {
                config.gate.mode = GateMode::Strict; changed = true;
            }
            if ui.selectable_label(config.gate.mode == GateMode::VadOnly, "VAD Only").clicked() {
                config.gate.mode = GateMode::VadOnly; changed = true;
            }
        });
        let mode_hint = match config.gate.mode {
            GateMode::Optimistic(_) => "Opens instantly, closes if not owner. Best for calls.",
            GateMode::Strict => "Stays closed until speaker verified. Best for recording.",
            GateMode::VadOnly => "Passes all speech, no speaker verification.",
        };
        ui.label(egui::RichText::new(mode_hint).small().weak());

        // Optimistic-specific settings
        if let GateMode::Optimistic(ref mut opt_cfg) = config.gate.mode {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Verification settle (ms):")
                    .on_hover_text("Grace period before trusting speaker verification.\nDuring this time the gate stays open regardless of result.");
                let mut settle = opt_cfg.verification_settle_ms as f32;
                if ui.add(egui::Slider::new(&mut settle, 0.0..=2000.0).step_by(50.0)).changed() {
                    opt_cfg.verification_settle_ms = settle as u32;
                    changed = true;
                }
            });
        }
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Audio Devices").strong());
        ui.label(
            egui::RichText::new("Changes take effect on next Start.")
                .small()
                .weak(),
        );
        ui.add_space(4.0);

        // Periodically refresh device lists in the background.
        devices.request_refresh(ctx);

        let input_devices = devices.input_devices();
        let output_devices = devices.output_devices();

        // Input device
        let current_input = config
            .audio
            .input_device
            .clone()
            .unwrap_or_else(|| "(System Default)".to_string());
        ui.horizontal(|ui| {
            ui.label("Input:");
            let combo = egui::ComboBox::from_id_salt("input_device").selected_text(&current_input);
            let response = combo.show_ui(ui, |ui| {
                if ui
                    .selectable_label(config.audio.input_device.is_none(), "(System Default)")
                    .clicked()
                {
                    config.audio.input_device = None;
                    changed = true;
                    device_changed = true;
                }
                for name in &input_devices {
                    let selected = config.audio.input_device.as_deref() == Some(name.as_str());
                    if ui.selectable_label(selected, name).clicked() {
                        config.audio.input_device = Some(name.clone());
                        changed = true;
                        device_changed = true;
                    }
                }
            });
            // Trigger refresh when dropdown is opened.
            if response.response.clicked() {
                devices.request_refresh(ctx);
            }
        });

        // Output device
        let current_output = config
            .audio
            .output_device
            .clone()
            .unwrap_or_else(|| "(System Default)".to_string());
        ui.horizontal(|ui| {
            ui.label("Output:");
            let combo =
                egui::ComboBox::from_id_salt("output_device").selected_text(&current_output);
            let response = combo.show_ui(ui, |ui| {
                if ui
                    .selectable_label(config.audio.output_device.is_none(), "(System Default)")
                    .clicked()
                {
                    config.audio.output_device = None;
                    changed = true;
                    device_changed = true;
                }
                for name in &output_devices {
                    let selected = config.audio.output_device.as_deref() == Some(name.as_str());
                    if ui.selectable_label(selected, name).clicked() {
                        config.audio.output_device = Some(name.clone());
                        changed = true;
                        device_changed = true;
                    }
                }
            });
            if response.response.clicked() {
                devices.request_refresh(ctx);
            }
        });
    });

    ui.add_space(8.0);
    ui.group(|ui| {
        ui.label(RichText::new("Runtime").strong());
        ui.label("Inference: ort (ONNX Runtime)");
        ui.label("Silero VAD: loaded from models/silero_vad.onnx");
        ui.label("ECAPA-TDNN: loaded from models/ecapa_tdnn.onnx");
    });

    SettingsResult {
        changed,
        device_changed,
    }
}
