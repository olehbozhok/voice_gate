//! Top-level application — eframe App implementation.

use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Sender};
use parking_lot::RwLock;

use crate::config::Config;
use crate::models::{self, DownloadProgress, ModelStatus};
use crate::pipeline::processor::{EnrollmentCommand, PipelineTelemetry, Processor};
use crate::pipeline::verifier::SpeakerVerifier;
use crate::speaker::embedding::EcapaTdnn;
use crate::speaker::profile::{ProfileStore, VoiceProfile};
use crate::ui::enrollment_view::EnrollmentViewState;
use crate::ui::model_setup_view::ModelSetupAction;
use crate::ui::ActiveView;
use crate::vad::silero::SileroVad;

#[derive(Clone, Copy)]
enum EnrollmentAction {
    None,
    Start,
    Finalize,
    Cancel,
}

/// Pipeline state.
enum PipelineState {
    Idle,
    /// Background thread is loading ML models.
    LoadingModels,
    /// Models loaded, pipeline running.
    Running {
        _input_stream: cpal::Stream,
        _output_stream: cpal::Stream,
        _processor_handle: JoinHandle<()>,
        _stop_signal: Sender<Vec<f32>>,
        enrollment_tx: Sender<EnrollmentCommand>,
    },
}

/// ML models loaded in background thread (all are Send).
struct LoadedModels {
    vad: SileroVad,
    verifier: SpeakerVerifier,
    enrollment_ecapa: EcapaTdnn,
}

pub struct VoiceGateApp {
    config: Arc<RwLock<Config>>,
    config_path: PathBuf,
    active_view: ActiveView,
    profile_store: Arc<RwLock<ProfileStore>>,
    telemetry: Arc<RwLock<PipelineTelemetry>>,
    pipeline: PipelineState,
    last_error: Option<String>,
    recording_flag: Arc<AtomicBool>,
    device_cache: crate::ui::settings_view::DeviceListCache,
    enrollment_view_state: EnrollmentViewState,
    /// Receives loaded models from background thread.
    models_rx: Option<crossbeam_channel::Receiver<Result<LoadedModels, String>>>,
    /// Model readiness status — checked at startup.
    model_status: ModelStatus,
    /// Shared download progress, updated by download thread.
    download_progress: Option<Arc<Mutex<DownloadProgress>>>,
}

impl VoiceGateApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let config_path = PathBuf::from("config.json");
        let config = Config::load(&config_path);
        let profile_store = ProfileStore::load(&config.profiles_dir);
        let model_status = models::check_models(&config.models_dir);

        Self {
            config: Arc::new(RwLock::new(config)),
            config_path,
            active_view: ActiveView::Main,
            profile_store: Arc::new(RwLock::new(profile_store)),
            telemetry: Arc::new(RwLock::new(PipelineTelemetry::default())),
            pipeline: PipelineState::Idle,
            last_error: None,
            recording_flag: Arc::new(AtomicBool::new(false)),
            device_cache: crate::ui::settings_view::DeviceListCache::new(),
            enrollment_view_state: EnrollmentViewState::default(),
            models_rx: None,
            model_status,
            download_progress: None,
        }
    }

    fn is_running(&self) -> bool {
        matches!(self.pipeline, PipelineState::Running { .. })
    }

    fn is_starting(&self) -> bool {
        matches!(self.pipeline, PipelineState::LoadingModels)
    }

    fn start(&mut self, ctx: &egui::Context) {
        if self.is_running() || self.is_starting() {
            return;
        }

        self.pipeline = PipelineState::LoadingModels;
        self.last_error = None;

        let (tx, rx) = bounded(1);
        self.models_rx = Some(rx);

        let profiles = self.profile_store.read().profiles().to_vec();
        let models_dir = self.config.read().models_dir.clone();
        let ctx = ctx.clone();

        std::thread::Builder::new()
            .name("model-loader".into())
            .spawn(move || {
                let result = Self::load_models(&models_dir, profiles);
                let _ = tx.send(result.map_err(|e| format!("{:#}", e)));
                ctx.request_repaint();
            })
            .expect("failed to spawn model-loader thread");
    }

    /// Load ML models — runs on background thread.
    fn load_models(
        models_dir: &std::path::Path,
        profiles: Vec<VoiceProfile>,
    ) -> anyhow::Result<LoadedModels> {
        log::info!("Loading models from {}", models_dir.display());
        let vad_path = models::silero_vad_path(models_dir);
        let ecapa_path = models::ecapa_tdnn_path(models_dir);

        let vad = SileroVad::new(&vad_path)?;
        let ecapa_for_verifier = EcapaTdnn::new(&ecapa_path)?;
        let verifier = SpeakerVerifier::spawn(ecapa_for_verifier, profiles);
        let enrollment_ecapa = EcapaTdnn::new(&ecapa_path)?;
        log::info!("Models loaded");
        Ok(LoadedModels {
            vad,
            verifier,
            enrollment_ecapa,
        })
    }

    /// Start downloading missing models in background.
    fn start_download(&mut self, ctx: &egui::Context) {
        let progress = Arc::new(Mutex::new(DownloadProgress {
            status: ModelStatus::Downloading {
                current_model: String::new(),
                progress: 0.0,
            },
        }));
        self.download_progress = Some(progress.clone());
        self.model_status = ModelStatus::Downloading {
            current_model: "Starting...".into(),
            progress: 0.0,
        };

        let models_dir = self.config.read().models_dir.clone();
        let ctx = ctx.clone();

        std::thread::Builder::new()
            .name("model-downloader".into())
            .spawn(move || {
                if let Err(e) = models::download_models(&models_dir, progress.clone()) {
                    let mut p = progress.lock().unwrap();
                    p.status = ModelStatus::Error(format!("{:#}", e));
                }
                ctx.request_repaint();
            })
            .expect("failed to spawn model-downloader thread");
    }

    /// Poll download progress and update model_status.
    fn poll_download(&mut self) {
        if let Some(ref progress) = self.download_progress {
            let p = progress.lock().unwrap();
            self.model_status = p.status.clone();
            if matches!(self.model_status, ModelStatus::DownloadComplete | ModelStatus::Error(_)) {
                drop(p);
                self.download_progress = None;
                if matches!(self.model_status, ModelStatus::DownloadComplete) {
                    // Re-check to transition to Ready.
                    let models_dir = self.config.read().models_dir.clone();
                    self.model_status = models::check_models(&models_dir);
                }
            }
        }
    }

    /// Check if models have finished loading, then start audio + processor.
    fn poll_startup(&mut self) {
        let rx = match &self.models_rx {
            Some(rx) => rx,
            None => return,
        };

        let result = match rx.try_recv() {
            Ok(r) => r,
            Err(crossbeam_channel::TryRecvError::Empty) => return,
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.pipeline = PipelineState::Idle;
                self.last_error = Some("Model loader thread crashed".into());
                self.models_rx = None;
                return;
            }
        };
        self.models_rx = None;

        match result {
            Ok(models) => match self.start_pipeline(models) {
                Ok(()) => log::info!("Pipeline started"),
                Err(e) => {
                    self.pipeline = PipelineState::Idle;
                    self.last_error = Some(format!("Start failed: {:#}", e));
                    log::error!("Pipeline start failed: {:#}", e);
                }
            },
            Err(e) => {
                self.pipeline = PipelineState::Idle;
                self.last_error = Some(format!("Model loading failed: {}", e));
                log::error!("Model loading failed: {}", e);
            }
        }
    }

    /// Start audio streams and processor — runs on UI thread (fast, no model loading).
    fn start_pipeline(&mut self, models: LoadedModels) -> anyhow::Result<()> {
        let cfg = self.config.read();

        let input_dev = match &cfg.audio.input_device {
            Some(name) => crate::audio::capture::find_input_device(name)?,
            None => crate::audio::capture::default_input_device()?,
        };
        let (audio_tx, audio_rx) = bounded::<Vec<f32>>(64);
        let input_stream = crate::audio::capture::start_capture(&input_dev, audio_tx.clone())?;

        let output_dev = match &cfg.audio.output_device {
            Some(name) => crate::audio::output::find_output_device(name)?,
            None => crate::audio::output::default_output_device()?,
        };
        let (output_tx, output_rx) = bounded::<Vec<f32>>(64);
        let output_stream = crate::audio::output::start_output(&output_dev, output_rx)?;

        let profile_store = self.profile_store.clone();
        drop(cfg);

        let (enrollment_tx, enrollment_rx) = bounded::<EnrollmentCommand>(8);

        let telemetry = self.telemetry.clone();
        let config = self.config.clone();
        let recording_flag = self.recording_flag.clone();

        let handle = std::thread::Builder::new()
            .name("voice-gate-processor".into())
            .spawn(move || {
                let mut proc = Processor::new(
                    config,
                    models.vad,
                    models.verifier,
                    models.enrollment_ecapa,
                    telemetry,
                    recording_flag,
                    enrollment_rx,
                    profile_store,
                );
                if let Err(e) = proc.run(audio_rx, output_tx) {
                    log::error!("Processor error: {:#}", e);
                }
            })?;

        self.pipeline = PipelineState::Running {
            _input_stream: input_stream,
            _output_stream: output_stream,
            _processor_handle: handle,
            _stop_signal: audio_tx,
            enrollment_tx,
        };
        Ok(())
    }

    fn stop(&mut self) {
        self.pipeline = PipelineState::Idle;
        *self.telemetry.write() = PipelineTelemetry::default();
        log::info!("Pipeline stopped");
    }

    fn toggle(&mut self, ctx: &egui::Context) {
        if self.is_running() {
            self.stop();
        } else {
            self.start(ctx);
        }
    }

    fn send_enrollment(&self, cmd: EnrollmentCommand) {
        if let PipelineState::Running {
            ref enrollment_tx, ..
        } = self.pipeline
        {
            let _ = enrollment_tx.try_send(cmd);
        }
    }
}

impl eframe::App for VoiceGateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_startup();
        self.poll_download();

        // If downloading, repaint to update progress bar.
        if self.download_progress.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // ── Model setup gate ──────────────────────────────────────────
        if !matches!(self.model_status, ModelStatus::Ready) {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            egui::CentralPanel::default().show(ctx, |ui| {
                let models_dir = self.config.read().models_dir.display().to_string();
                let action = crate::ui::model_setup_view::show(
                    ui,
                    &self.model_status,
                    &models_dir,
                    &self.download_progress,
                );
                match action {
                    ModelSetupAction::None => {}
                    ModelSetupAction::Download => self.start_download(ctx),
                    ModelSetupAction::SetModelsDir(path) => {
                        self.config.write().models_dir = path;
                        let _ = self.config.read().save(&self.config_path);
                        let models_dir = self.config.read().models_dir.clone();
                        self.model_status = models::check_models(&models_dir);
                    }
                }
            });
            return;
        }

        if self.is_running() || self.is_starting() {
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
                    ui.label(
                        egui::RichText::new(format!("Error: {}", err))
                            .color(egui::Color32::from_rgb(220, 60, 60)),
                    );
                    if ui.small_button("x").clicked() {
                        clear_error = true;
                    }
                });
            });
        }
        if clear_error {
            self.last_error = None;
        }

        // Central panel
        let ctx_clone = ctx.clone();
        egui::CentralPanel::default().show(ctx, |ui| match self.active_view {
            ActiveView::Main => {
                if self.is_starting() {
                    ui.heading("Voice Gate");
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Loading models...");
                    });
                    return;
                }

                let telem = self.telemetry.clone();
                let cfg = self.config.clone();
                let running = self.is_running();
                let has_profile = !self.profile_store.read().is_empty();
                let is_recording = self
                    .recording_flag
                    .load(std::sync::atomic::Ordering::Relaxed);
                let flag = self.recording_flag.clone();

                crate::ui::main_view::show(
                    ui,
                    &telem,
                    &cfg,
                    running,
                    has_profile,
                    &mut || self.toggle(&ctx_clone),
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

                if !self.is_running() {
                    ui.heading("Voice Enrollment");
                    ui.add_space(8.0);
                    ui.label("Start the pipeline first (click Start on Dashboard).");
                    return;
                }

                let action = Cell::new(EnrollmentAction::None);
                crate::ui::enrollment_view::show(
                    ui,
                    &state,
                    secs,
                    min,
                    &self.profile_store,
                    &mut self.enrollment_view_state,
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
        });
    }
}
