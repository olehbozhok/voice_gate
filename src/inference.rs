//! ONNX inference wrapper — hides ort (ONNX Runtime) behind a stable API.
//!
//! To swap the inference backend, change only this file.
//! No other module imports ort types directly.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::Session;

/// Opaque wrapper over an ONNX Runtime session.
pub struct OnnxModel {
    session: Session,
}

/// Opaque handle for stateful model data (e.g. LSTM hidden/cell state).
/// Stores an N-dimensional array with its shape preserved for round-tripping.
pub struct ModelState {
    data: Vec<f32>,
    shape: Vec<usize>,
}

/// Typed model input — no framework-specific types exposed.
pub enum Input {
    /// Dense f32 tensor with explicit shape.
    F32 { shape: Vec<usize>, data: Vec<f32> },
    /// Dense i64 tensor with explicit shape.
    I64 { shape: Vec<usize>, data: Vec<i64> },
    /// Opaque state from a previous inference call.
    State(ModelState),
}

/// Opaque model output with typed accessors.
pub struct Output {
    data: Vec<f32>,
}

// ── OnnxModel ────────────────────────────────────────────────────────────

impl OnnxModel {
    /// Load an ONNX model from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create ONNX Runtime session builder")?
            .commit_from_file(path)
            .with_context(|| format!("failed to load ONNX model: {}", path.display()))?;
        Ok(Self { session })
    }

    /// Run inference with the given inputs.
    pub fn run(&mut self, inputs: Vec<Input>) -> Result<Vec<Output>> {
        let ort_inputs: Vec<ort::session::SessionInputValue<'_>> = inputs
            .into_iter()
            .map(|inp| inp.into_session_input())
            .collect::<Result<_>>()?;

        let outputs = self
            .session
            .run(ort_inputs.as_slice())
            .context("inference failed")?;

        outputs
            .values()
            .map(|value| {
                let (_shape, data) = value
                    .try_extract_tensor::<f32>()
                    .context("failed to extract f32 tensor from output")?;
                let data: Vec<f32> = data.to_vec();
                Ok(Output { data })
            })
            .collect()
    }
}

// ── Input ────────────────────────────────────────────────────────────────

impl Input {
    fn into_session_input(self) -> Result<ort::session::SessionInputValue<'static>> {
        match self {
            Input::F32 { shape, data } => {
                let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                let value = ort::value::Tensor::from_array((shape_i64, data.into_boxed_slice()))
                    .context("failed to create f32 tensor")?;
                Ok(value.into())
            }
            Input::I64 { shape, data } => {
                let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
                let value = ort::value::Tensor::from_array((shape_i64, data.into_boxed_slice()))
                    .context("failed to create i64 tensor")?;
                Ok(value.into())
            }
            Input::State(state) => {
                let shape_i64: Vec<i64> = state.shape.iter().map(|&d| d as i64).collect();
                let value =
                    ort::value::Tensor::from_array((shape_i64, state.data.into_boxed_slice()))
                        .context("failed to create state tensor")?;
                Ok(value.into())
            }
        }
    }
}

// ── Output ───────────────────────────────────────────────────────────────

impl Output {
    /// Extract the output as a flat Vec<f32>.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }
}

// ── ModelState ───────────────────────────────────────────────────────────

impl ModelState {
    /// Create a state tensor from existing data with explicit shape.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Create a zero-filled f32 state tensor with the given shape.
    pub fn zeros_f32(shape: &[usize]) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape: shape.to_vec(),
        }
    }
}
