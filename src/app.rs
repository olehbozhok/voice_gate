//! Top-level application — eframe App implementation.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::pipeline::processor::{PipelineTelemetry, Processor};
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::enrollment::{EnrollmentSession, EnrollmentState};
use crate::speaker::profile::VoiceProfile;
use crate::ui::ActiveView;
use crate::vad::silero::SileroVad;

#[derive(Clone, Copy)]
enum EnrollmentAction { None, Start, Finalize, Reset }

struct LivePipeline {
    _input_stream: cpal::Stream,
    _output_stream: cpal::Stream,
    _processor_handle: JoinHandle<()>,
    _stop_signal: Sender<Vec<f32>>,
}

pub struct VoiceGateApp {
    config: Config,
    config_path: PathBuf,
    active_view: ActiveView,
    is_running: bool,
    voice_profile: Option<VoiceProfile>,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    live: Option<LivePipeline>,
    enrollment: Option<EnrollmentSession>,
    last_error: Option<String>,
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
            config, config_path,
            active_view: ActiveView::Main,
            is_running: false, voice_profile,
            telemetry: Arc::new(RwLock::new(PipelineTelemetry::default())),
            live: None, enrollment: None, last_error: None,
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
        let sr = self.config.audio.sample_rate;
        let fs = self.config.audio.frame_samples;

        // Audio I/O
        let input_dev = crate::audio::capture::default_input_device()?;
        let (audio_tx, audio_rx) = bounded::<Vec<f32>>(64);
        let (input_stream, _) = crate::audio::capture::start_capture(&input_dev, sr, fs, audio_tx.clone())?;

        let output_dev = crate::audio::output::default_output_device()?;
        let (output_tx, output_rx) = bounded::<Vec<f32>>(64);
        let output_stream = crate::audio::output::start_output(&output_dev, sr, output_rx)?;

        // ML Models (tract — loaded from ONNX at runtime)
        let vad = SileroVad::new(
            self.config.vad.threshold,
            Path::new("models/silero_vad.onnx"),
        )?;
        let ecapa = EcapaTdnn::new(
            Path::new("models/ecapa_tdnn.onnx"),
        )?;

        // Processor thread
        let telemetry = self.telemetry.clone();
        let profile = self.voice_profile.clone();
        let config = self.config.clone();

        let handle = std::thread::Builder::new()
            .name("voice-gate-processor".into())
            .spawn(move || {
                let mut proc = Processor::new(config, vad, ecapa, profile, telemetry);
                if let Err(e) = proc.run(audio_rx, output_tx) {
                    log::error!("Processor error: {:#}", e);
                }
            })?;

        self.live = Some(LivePipeline {
            _input_stream: input_stream, _output_stream: output_stream,
            _processor_handle: handle, _stop_signal: audio_tx,
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
}

impl eframe::App for VoiceGateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running { ctx.request_repaint(); }

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
                    crate::ui::main_view::show(ui, &telem, running, has_profile, &mut || self.toggle());
                }
                ActiveView::Enrollment => {
                    if self.enrollment.is_none() {
                        self.enrollment = Some(EnrollmentSession::new(
                            self.config.audio.sample_rate, self.config.speaker.min_enrollment_seconds,
                        ));
                    }
                    let e = self.enrollment.as_ref().unwrap();
                    let state = e.state.clone();
                    let secs = e.speech_seconds();
                    let min = self.config.speaker.min_enrollment_seconds;
                    let action = Cell::new(EnrollmentAction::None);
                    crate::ui::enrollment_view::show(ui, &state, secs, min,
                        &mut || action.set(EnrollmentAction::Start),
                        &mut || action.set(EnrollmentAction::Finalize),
                        &mut || action.set(EnrollmentAction::Reset),
                    );
                    let e = self.enrollment.as_mut().unwrap();
                    match action.get() {
                        EnrollmentAction::None => {}
                        EnrollmentAction::Start => e.start(),
                        EnrollmentAction::Finalize => { e.state = EnrollmentState::Processing; }
                        EnrollmentAction::Reset => e.reset(),
                    }
                }
                ActiveView::Settings => {
                    if crate::ui::settings_view::show(ui, &mut self.config) {
                        let _ = self.config.save(&self.config_path);
                    }
                }
            }
        });
    }
}
