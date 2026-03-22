//! Backend selection for Burn inference.
//!
//! Burn is generic over its `Backend` trait, which means the same model code
//! can run on CPU, GPU (Vulkan/Metal/DX12), or CUDA without any changes.
//! This module defines a type alias `AppBackend` that resolves to the correct
//! backend based on feature flags.
//!
//! ```bash
//! cargo build --release                 # → NdArray (CPU, pure Rust)
//! cargo build --release --features wgpu # → Wgpu (GPU via Vulkan/Metal/DX12)
//! cargo build --release --features cuda # → CUDA (NVIDIA GPU)
//! ```

use burn::backend;

// ── Compile-time validation ─────────────────────────────────────────────────

// Ensure at least one backend is selected.
#[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "cuda")))]
compile_error!(
    "\n\n\
    ========================================\n\
    voice-gate: no inference backend selected\n\
    ========================================\n\n\
    You must enable at least one backend feature:\n\n\
      cargo build --release                   # CPU (default)\n\
      cargo build --release --features wgpu   # GPU (Vulkan/Metal/DX12)\n\
      cargo build --release --features cuda   # NVIDIA CUDA\n\n"
);

// Warn if multiple backends are active (CUDA wins, which is fine, but
// the user should know what they're getting).
#[cfg(all(feature = "cuda", feature = "wgpu"))]
compile_error!(
    "\n\n\
    ========================================\n\
    voice-gate: conflicting backends\n\
    ========================================\n\n\
    Both `cuda` and `wgpu` features are enabled. Pick one:\n\n\
      --features cuda   # NVIDIA CUDA\n\
      --features wgpu   # Vulkan/Metal/DX12 (works on all GPUs)\n\n"
);

/// The concrete Burn backend, selected at compile time via feature flags.
#[cfg(feature = "cuda")]
pub type AppBackend = backend::Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type AppBackend = backend::Wgpu;

#[cfg(all(feature = "cpu", not(feature = "wgpu"), not(feature = "cuda")))]
pub type AppBackend = backend::NdArray;

/// Human-readable name of the active backend.
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    { "CUDA" }

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    { "WGPU (Vulkan/Metal/DX12)" }

    #[cfg(all(feature = "cpu", not(feature = "wgpu"), not(feature = "cuda")))]
    { "NdArray (CPU)" }
}
