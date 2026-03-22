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

/// Element type for input fact declarations.
#[derive(Debug, Clone, Copy)]
pub enum DType {
    F32,
    I64,
}

/// Describes the expected shape, type, or constant value of a model input.
/// Used to resolve dynamic shapes and conditional branches before optimization.
pub enum InputFact {
    /// Input with a known shape and type. Use `0` in shape for dynamic dimensions.
    Shape { shape: Vec<usize>, dtype: DType },
    /// Input with a fixed constant value, baked into the model at load time.
    /// Enables the optimizer to resolve conditional branches (e.g. If nodes).
    ConstI64(i64),
}

/// Opaque model output with typed accessors.
pub struct Output {
    inner: Tensor,
}

// ── OnnxModel ────────────────────────────────────────────────────────────

impl OnnxModel {
    /// Load an ONNX model from disk, optimize the graph, and prepare for
    /// inference. Use this when the model has fully determined input shapes.
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_inputs(path, &[])
    }

    /// Load an ONNX model and set explicit input facts for models with
    /// dynamic or undetermined input shapes. Each `InputFact` specifies the
    /// shape and element type of the corresponding model input (by position).
    /// A dimension of `0` in the shape means "dynamic" (variable length).
    pub fn load_with_inputs(path: &Path, input_facts: &[InputFact]) -> Result<Self> {
        let mut model = tract_onnx::onnx()
            .model_for_path(path)
            .with_context(|| format!("failed to load ONNX model: {}", path.display()))?;

        let symbols = SymbolScope::default();
        for (i, fact) in input_facts.iter().enumerate() {
            match fact {
                InputFact::Shape { shape, dtype } => {
                    let dims: Vec<TDim> = shape.iter().enumerate().map(|(dim_idx, &d)| {
                        if d == 0 {
                            let name = format!("d{}_{}", i, dim_idx);
                            symbols.sym(&name).into()
                        } else {
                            d.into()
                        }
                    }).collect();
                    let dt = match dtype {
                        DType::F32 => f32::datum_type(),
                        DType::I64 => i64::datum_type(),
                    };
                    model.set_input_fact(i, InferenceFact::dt_shape(dt, &dims))?;
                }
                InputFact::ConstI64(value) => {
                    let tensor = Tensor::from(tract_ndarray::arr0(*value));
                    model.set_input_fact(i, InferenceFact::from(tensor))?;
                }
            }
        }

        let plan = model
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
