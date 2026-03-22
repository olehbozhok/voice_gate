//! Gate state machine — determines whether audio passes through.
//!
//! States: Silent → MyVoice (open) / OtherVoice (closed) → Trailing → Silent

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Silent,
    MyVoice,
    OtherVoice,
    Trailing,
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
