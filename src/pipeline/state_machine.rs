//! Gate state machine — determines whether audio passes through.
//!
//! States: Silent → MyVoice (open) / OtherVoice (closed) → Trailing → Silent

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Silent,
    MyVoice,
    OtherVoice,
    Trailing,
}

impl GateState {
    pub fn is_open(self) -> bool {
        matches!(self, GateState::MyVoice | GateState::Trailing)
    }
}

impl std::fmt::Display for GateState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateState::Silent => write!(f, "Silent"),
            GateState::MyVoice => write!(f, "My Voice"),
            GateState::OtherVoice => write!(f, "Other Voice"),
            GateState::Trailing => write!(f, "Trailing"),
        }
    }
}

pub struct GateStateMachine {
    state: GateState,
    trailing_since: Option<Instant>,
    hold_duration: Duration,
}

impl GateStateMachine {
    pub fn new(hold_time_ms: u32) -> Self {
        Self { state: GateState::Silent, trailing_since: None, hold_duration: Duration::from_millis(hold_time_ms as u64) }
    }

    pub fn state(&self) -> GateState { self.state }
    pub fn is_open(&self) -> bool { self.state.is_open() }

    /// Advance one frame. `is_owner` only meaningful when `is_speech` is true.
    pub fn update(&mut self, is_speech: bool, is_owner: bool) -> GateState {
        self.state = match self.state {
            GateState::Silent => {
                if is_speech && is_owner { GateState::MyVoice }
                else if is_speech { GateState::OtherVoice }
                else { GateState::Silent }
            }
            GateState::MyVoice => {
                if is_speech && is_owner { GateState::MyVoice }
                else if is_speech { GateState::OtherVoice }
                else { self.trailing_since = Some(Instant::now()); GateState::Trailing }
            }
            GateState::OtherVoice => {
                if !is_speech { GateState::Silent }
                else if is_owner { GateState::MyVoice }
                else { GateState::OtherVoice }
            }
            GateState::Trailing => {
                if is_speech && is_owner { self.trailing_since = None; GateState::MyVoice }
                else if let Some(since) = self.trailing_since {
                    if since.elapsed() >= self.hold_duration { self.trailing_since = None; GateState::Silent }
                    else { GateState::Trailing }
                } else { GateState::Silent }
            }
        };
        self.state
    }

    pub fn set_hold_time(&mut self, ms: u32) { self.hold_duration = Duration::from_millis(ms as u64); }
    pub fn reset(&mut self) { self.state = GateState::Silent; self.trailing_since = None; }
}
