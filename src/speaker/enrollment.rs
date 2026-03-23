//! Voice enrollment — records user speech and builds a VoiceProfile.

use crate::vad::VadResult;

/// Minimum speech segment duration to keep. Shorter segments
/// (coughs, breaths, single words) are discarded as noise.
const MIN_SEGMENT_SECS: f32 = 0.35;

#[derive(Debug, Clone, PartialEq)]
pub enum EnrollmentState {
    Idle,
    Recording { speech_seconds: f32 },
    Processing,
    Done,
    Failed(String),
}

/// Accumulates speech audio during enrollment and produces embedding windows.
pub struct EnrollmentSession {
    sample_rate: u32,
    speech_buffer: Vec<f32>,
    current_segment: Vec<f32>,
    prev_was_speech: bool,
    pub state: EnrollmentState,
}

impl EnrollmentSession {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            speech_buffer: Vec::new(),
            current_segment: Vec::new(),
            prev_was_speech: false,
            state: EnrollmentState::Idle,
        }
    }

    pub fn start(&mut self) {
        self.speech_buffer.clear();
        self.current_segment.clear();
        self.prev_was_speech = false;
        self.state = EnrollmentState::Recording {
            speech_seconds: 0.0,
        };
    }

    /// Feed a frame + VAD result. Call every frame while Recording.
    pub fn feed_frame(&mut self, frame: &[f32], vad: &VadResult) {
        if !matches!(self.state, EnrollmentState::Recording { .. }) {
            return;
        }
        if vad.is_speech {
            self.current_segment.extend_from_slice(frame);
            self.prev_was_speech = true;
        } else if self.prev_was_speech {
            let secs = self.current_segment.len() as f32 / self.sample_rate as f32;
            if secs >= MIN_SEGMENT_SECS {
                self.speech_buffer.extend_from_slice(&self.current_segment);
            }
            self.current_segment.clear();
            self.prev_was_speech = false;
        }
        let speech_seconds = self.speech_seconds();
        self.state = EnrollmentState::Recording { speech_seconds };
    }

    /// Total accumulated speech including the segment currently being recorded.
    pub fn speech_seconds(&self) -> f32 {
        (self.speech_buffer.len() + self.current_segment.len()) as f32 / self.sample_rate as f32
    }

    /// Extract overlapping windows from accumulated speech for embedding.
    /// Returns Vec of audio chunks, each ~3 seconds.
    pub fn get_embedding_windows(&self) -> Vec<Vec<f32>> {
        let window = (3.0 * self.sample_rate as f32) as usize;
        let hop = (1.5 * self.sample_rate as f32) as usize;
        let mut windows = Vec::new();
        let mut offset = 0;
        while offset + window <= self.speech_buffer.len() {
            windows.push(self.speech_buffer[offset..offset + window].to_vec());
            offset += hop;
        }
        if offset < self.speech_buffer.len() {
            let remaining = &self.speech_buffer[offset..];
            if remaining.len() >= self.sample_rate as usize {
                windows.push(remaining.to_vec());
            }
        }
        windows
    }
}
