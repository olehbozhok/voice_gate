//! Top-level application — eframe App implementation.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::pipeline::processor::{EnrollmentCommand, PipelineTelemetry, Processor};
use crate::pipeline::verifier::SpeakerVerifier;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::VoiceProfile;
use crate::ui::ActiveView;
use crate::vad::silero::SileroVad;

#[derive(Clone, Copy)]
enum EnrollmentAction { None, Start, Finalize, Cancel }

struct LivePipeline {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    _processor_handle: JoinHandle<()>,
    _stop_signal: Sender<Vec<f32>>,
    enrollment_tx: Sender<EnrollmentCommand>,
}

pub struct VoiceGateApp {
    config: Arc<RwLock<Config>>,
    config_path: PathBuf,
    active_view: ActiveView,
    is_running: bool,
    voice_profile: Option<VoiceProfile>,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    live: Option<LivePipeline>,
    last_error: Option<String>,
    recording_flag: Arc<AtomicBool>,
    device_cache: crate::ui::settings_view::DeviceListCache,
}

impl VoiceGateApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let config_path = PathBuf::from("config.json");
        let config = Config::load(&config_path);
        let profile_path = config.profiles_dir.join("default.json");
        let voice_profile = VoiceProfile::load(&profile_path).ok();

        if voice_profile.is_some() { log::info!("Loaded voice profile"); }
        else { log::info!("No voice profile — enrollment required"); }

        Self {
            config: Arc::new(RwLock::new(config)),
            config_path,
            active_view: ActiveView::Main,
            is_running: false, voice_profile,
            telemetry: Arc::new(RwLock::new(PipelineTelemetry::default())),
            live: None, last_error: None,
            recording_flag: Arc::new(AtomicBool::new(false)),
            device_cache: crate::ui::settings_view::DeviceListCache::new(),
        }
    }

    fn start(&mut self) {
        if self.is_running { return; }
        match self.try_start() {
            Ok(()) => { self.is_running = true; self.last_error = None; log::info!("Pipeline started"); }
            Err(e) => { self.last_error = Some(format!("Start failed: {}", e)); log::error!("{:#}", e); }
        }
    }

    fn try_start(&mut self) -> anyhow::Result<()> {
        let cfg = self.config.read();

        // Input device
        let input_dev = match &cfg.audio.input_device {
            Some(name) => crate::audio::capture::find_input_device(name)?,
            None => crate::audio::capture::default_input_device()?,
        };
        let (audio_tx, audio_rx) = bounded::<Vec<f32>>(64);
        let input_stream = crate::audio::capture::start_capture(&input_dev, audio_tx.clone())?;

        // Output device
        let output_dev = match &cfg.audio.output_device {
            Some(name) => crate::audio::output::find_output_device(name)?,
            None => crate::audio::output::default_output_device()?,
        };
        let (output_tx, output_rx) = bounded::<Vec<f32>>(64);
        let output_stream = crate::audio::output::start_output(&output_dev, output_rx)?;

        // ML Models (ort — loaded from ONNX at runtime)
        let vad = SileroVad::new(Path::new("models/silero_vad.onnx"))?;

        // Speaker verification — runs ECAPA-TDNN in a background thread
        let ecapa_for_verifier = EcapaTdnn::new(Path::new("models/ecapa_tdnn.onnx"))?;
        let verifier = SpeakerVerifier::spawn(
            ecapa_for_verifier,
            self.voice_profile.clone(),
            self.config.clone(),
        );

        // Separate ECAPA-TDNN instance for enrollment (one-time use)
        let ecapa_for_enrollment = EcapaTdnn::new(Path::new("models/ecapa_tdnn.onnx"))?;

        let profile_dir = cfg.profiles_dir.clone();
        drop(cfg); // release read lock before spawning

        // Enrollment command channel
        let (enrollment_tx, enrollment_rx) = bounded::<EnrollmentCommand>(8);

        // Processor thread
        let telemetry = self.telemetry.clone();
        let config = self.config.clone();
        let recording_flag = self.recording_flag.clone();

        let handle = std::thread::Builder::new()
            .name("voice-gate-processor".into())
            .spawn(move || {
                let mut proc = Processor::new(
                    config, vad, verifier, ecapa_for_enrollment,
                    telemetry, recording_flag, enrollment_rx, profile_dir,
                );
                if let Err(e) = proc.run(audio_rx, output_tx) {
                    log::error!("Processor error: {:#}", e);
                }
            })?;

        self.live = Some(LivePipeline {
            _input_stream: input_stream, _output_stream: output_stream,
            _processor_handle: handle, _stop_signal: audio_tx,
            enrollment_tx,
        });
        Ok(())
    }

    fn stop(&mut self) {
        self.live = None;
        self.is_running = false;
        *self.telemetry.write() = PipelineTelemetry::default();
        log::info!("Pipeline stopped");
    }

    fn toggle(&mut self) {
        if self.is_running { self.stop(); } else { self.start(); }
    }

    /// Send an enrollment command to the processor thread.
    fn send_enrollment(&self, cmd: EnrollmentCommand) {
        if let Some(ref live) = self.live {
            let _ = live.enrollment_tx.try_send(cmd);
        }
    }
}

impl eframe::App for VoiceGateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(33));
        }

        // Nav bar
        egui::TopBottomPanel::top("nav").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_view, ActiveView::Main, "Dashboard");
                ui.selectable_value(&mut self.active_view, ActiveView::Enrollment, "Enrollment");
                ui.selectable_value(&mut self.active_view, ActiveView::Settings, "Settings");
            });
        });

        // Error banner
        let mut clear_error = false;
        if let Some(err) = &self.last_error {
            let err = err.clone();
            egui::TopBottomPanel::top("error").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("Error: {}", err)).color(egui::Color32::from_rgb(220, 60, 60)));
                    if ui.small_button("x").clicked() { clear_error = true; }
                });
            });
        }
        if clear_error { self.last_error = None; }

        // Central panel
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.active_view {
                ActiveView::Main => {
                    let telem = self.telemetry.clone();
                    let running = self.is_running;
                    let has_profile = self.voice_profile.is_some();
                    let is_recording = self.recording_flag.load(std::sync::atomic::Ordering::Relaxed);
                    let flag = self.recording_flag.clone();
                    crate::ui::main_view::show(
                        ui, &telem, running, has_profile,
                        &mut || self.toggle(),
                        is_recording,
                        &mut || {
                            let prev = flag.load(std::sync::atomic::Ordering::Relaxed);
                            flag.store(!prev, std::sync::atomic::Ordering::Relaxed);
                        },
                    );
                }
                ActiveView::Enrollment => {
                    let t = self.telemetry.read();
                    let state = t.enrollment_state.clone();
                    let secs = t.enrollment_speech_secs;
                    drop(t);
                    let min = self.config.read().speaker.min_enrollment_seconds;

                    if !self.is_running {
                        ui.heading("Voice Enrollment");
                        ui.add_space(8.0);
                        ui.label("Start the pipeline first (click Start on Dashboard).");
                        return;
                    }

                    let action = Cell::new(EnrollmentAction::None);
                    crate::ui::enrollment_view::show(ui, &state, secs, min,
                        &mut || action.set(EnrollmentAction::Start),
                        &mut || action.set(EnrollmentAction::Finalize),
                        &mut || action.set(EnrollmentAction::Cancel),
                    );
                    match action.get() {
                        EnrollmentAction::None => {}
                        EnrollmentAction::Start => self.send_enrollment(EnrollmentCommand::Start),
                        EnrollmentAction::Finalize => self.send_enrollment(EnrollmentCommand::Finalize),
                        EnrollmentAction::Cancel => self.send_enrollment(EnrollmentCommand::Cancel),
                    }
                }
                ActiveView::Settings => {
                    let mut cfg = self.config.write();
                    if crate::ui::settings_view::show(ui, &mut cfg, &self.device_cache, ctx) {
                        let _ = cfg.save(&self.config_path);
                    }
                }
            }
        });
    }
}
