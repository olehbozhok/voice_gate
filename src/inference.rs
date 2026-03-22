//! ONNX inference wrapper — hides the tract runtime behind a stable API.
//!
//! To swap the inference backend (e.g. to ort), change only this file.
//! No other module imports tract types directly.

use std::path::Path;

use anyhow::{Context, Result};
use tract_onnx::prelude::*;

/// Opaque wrapper over an optimized ONNX model.
pub struct OnnxModel {
    plan: TypedSimplePlan<TypedModel>,
}

/// Opaque handle for stateful model data (e.g. LSTM hidden/cell state).
/// Passed back into subsequent inference calls without copying.
pub struct ModelState {
    inner: Tensor,
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
    inner: Tensor,
}

// ── OnnxModel ────────────────────────────────────────────────────────────

impl OnnxModel {
    /// Load an ONNX model from disk, optimize the graph, and prepare for
    /// inference.
    pub fn load(path: &Path) -> Result<Self> {
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("failed to load ONNX model: {}", path.display()))?
            .into_optimized()
            .with_context(|| format!("failed to optimize model: {}", path.display()))?
            .into_runnable()
            .with_context(|| format!("failed to prepare model: {}", path.display()))?;
        Ok(Self { plan })
    }

    /// Run inference with the given inputs.
    pub fn run(&self, inputs: Vec<Input>) -> Result<Vec<Output>> {
        let tv: TVec<TValue> = inputs
            .into_iter()
            .map(|inp| inp.into_tvalue())
            .collect::<Result<_>>()?;
        let outputs = self.plan.run(tv)?;
        Ok(outputs
            .into_iter()
            .map(|v: TValue| Output { inner: v.into_tensor() })
            .collect())
    }
}

// ── Input ────────────────────────────────────────────────────────────────

impl Input {
    fn into_tvalue(self) -> Result<TValue> {
        match self {
            Input::F32 { shape, data } => {
                let tensor = tract_ndarray::Array::from_shape_vec(
                    tract_ndarray::IxDyn(&shape),
                    data,
                )?;
                Ok(Tensor::from(tensor).into())
            }
            Input::I64 { shape, data } => {
                let tensor = tract_ndarray::Array::from_shape_vec(
                    tract_ndarray::IxDyn(&shape),
                    data,
                )?;
                Ok(Tensor::from(tensor).into())
            }
            Input::State(state) => Ok(state.inner.into()),
        }
    }
}

// ── Output ───────────────────────────────────────────────────────────────

impl Output {
    /// Extract a single f32 scalar from the output.
    pub fn to_scalar_f32(&self) -> Result<f32> {
        Ok(*self.inner.to_scalar::<f32>()?)
    }

    /// Extract the output as a flat Vec<f32>.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        Ok(self.inner.as_slice::<f32>()?.to_vec())
    }

    /// Convert this output into an opaque ModelState for round-tripping.
    pub fn into_state(self) -> ModelState {
        ModelState { inner: self.inner }
    }
}

// ── ModelState ───────────────────────────────────────────────────────────

impl ModelState {
    /// Create a zero-filled f32 state tensor with the given shape.
    pub fn zeros_f32(shape: &[usize]) -> Self {
        let tensor = tract_ndarray::Array::<f32, _>::zeros(
            tract_ndarray::IxDyn(shape),
        );
        Self {
            inner: Tensor::from(tensor),
        }
    }
}
