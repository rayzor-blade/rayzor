//! Rayzor GPU Compute — opt-in native package
//!
//! Provides GPU-accelerated compute via Metal (macOS), with CUDA and WebGPU
//! planned for future phases. Ships as a cdylib loaded at runtime via dlopen.
//!
//! # Plugin Registration
//!
//! Method descriptors are declared via [`declare_native_methods!`] and exported
//! through `rayzor_gpu_plugin_describe()`. The compiler reads these at load time
//! to auto-register method mappings and extern declarations — **no compiler core
//! changes required**.

#![recursion_limit = "256"]
// All extern "C" functions in this crate are FFI entry points called by the JIT runtime.
#![allow(clippy::missing_safety_doc)]
// Doc comments use prose with shape-indexing (`[b,m,n]`) and Haxe-side
// type shorthand (`Map<T>`) that rustdoc treats as broken links / unclosed
// HTML; silence rather than reformat every prose mention.
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(rustdoc::invalid_html_tags)]

// Compute modules — core logic uses buffer/lazy/backend types (native-only)
#[cfg(feature = "native")]
pub mod codegen;
#[cfg(feature = "native")]
pub mod kernel_cache;
#[cfg(feature = "native")]
pub mod kernel_ir;

// These modules have extern "C" FFI mixed with core logic —
// gate them on native, provide wasm alternatives via wasm_exports.rs
#[cfg(feature = "native")]
pub mod buffer;
#[cfg(all(feature = "native", feature = "webgpu-backend"))]
pub use buffer::GpuBuffer;
#[cfg(feature = "native")]
pub mod device;
#[cfg(feature = "native")]
pub mod lazy;
#[cfg(feature = "native")]
pub mod ops;
#[cfg(any(feature = "native", feature = "webgpu-backend"))]
pub mod xgpu_runtime;

// xgpu supplies the portable WebGPU object model and backend. Rayzor owns the
// ABI carriers above and layers its shader compiler, lazy compute graph and
// native kernels over the same handles through `xgpu_backend::extension`.
#[cfg(feature = "webgpu-backend")]
mod handles {
    pub use xgpu_core::{Slab, kind_of};
}
#[cfg(feature = "webgpu-backend")]
mod types {
    pub use xgpu_core::Kind;
}
#[cfg(feature = "webgpu-backend")]
mod runtime {
    pub use crate::xgpu_runtime::{
        Buffer, BufferMut, ErrorKind, Future, Rooted, Text, Value, host,
    };
}
#[cfg(feature = "webgpu-backend")]
#[allow(clippy::all)]
mod xgpu_backend {
    include!(concat!(env!("OUT_DIR"), "/xgpu_backend/backend.rs"));
}
#[cfg(feature = "webgpu-backend")]
#[allow(clippy::all)]
mod xgpu_api {
    #![allow(non_snake_case)]
    #[cfg(feature = "native")]
    use crate::GpuBuffer;
    use crate::xgpu_backend as backend;
    use crate::xgpu_runtime::{
        Buffer, BufferMut, Enum, ErrorKind, Future, NativeEnum, Rooted, Text, host,
    };
    include!(concat!(env!("OUT_DIR"), "/xgpu_rayzor.rs"));
}
#[cfg(feature = "webgpu-backend")]
pub use xgpu_api::*;

#[cfg(feature = "native")]
pub mod backend;

#[cfg(feature = "metal-backend")]
pub mod metal;

#[cfg(any(feature = "webgpu-backend", feature = "wasm-host"))]
pub mod wgpu_backend;

#[cfg(any(feature = "webgpu-backend", feature = "wasm-host"))]
pub mod graphics;

#[cfg(feature = "cuda-backend")]
pub mod cuda;

#[cfg(feature = "wasm-host")]
pub mod wasm_exports;

#[cfg(feature = "native")]
use rayzor_plugin::{NativeMethodDesc, declare_native_methods};

// The loader refuses a plugin with no `rayzor_plugin_abi_version` export
// (treated as ABI v0). This crate predates the handshake, so it failed to load
// even though its symbols were fine; the other plugins declare it the same way.
#[cfg(feature = "native")]
rayzor_plugin::export_abi_version!();

// ============================================================================
// Method descriptor table (read by compiler at plugin load time)
// Only compiled for native targets (not wasm-host)
#[cfg(feature = "native")]
// ============================================================================

declare_native_methods! {
    GPU_METHODS;
    // GPUCompute lifecycle (static)
    "rayzor::gpu::GPUCompute", "create",       static,   "rayzor_gpu_compute_create",        []              => Ptr;
    "rayzor::gpu::GPUCompute", "fromDevice",   static,   "rayzor_gpu_compute_from_device",   [Ptr]           => Ptr;
    "rayzor::gpu::GPUCompute", "isAvailable",  static,   "rayzor_gpu_compute_is_available",  []              => Bool;
    // GPUCompute instance methods (self = Ptr is first param)
    "rayzor::gpu::GPUCompute", "destroy",      instance, "rayzor_gpu_compute_destroy",       [Ptr]           => Void;
    "rayzor::gpu::GPUCompute", "createBuffer", instance, "rayzor_gpu_compute_create_buffer", [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "allocBuffer",  instance, "rayzor_gpu_compute_alloc_buffer",  [Ptr, I64, I64] => Ptr;
    "rayzor::gpu::GPUCompute", "toTensor",     instance, "rayzor_gpu_compute_to_tensor",     [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "freeBuffer",   instance, "rayzor_gpu_compute_free_buffer",   [Ptr, Ptr]      => Void;
    // Binary elementwise ops: (self, a, b) -> result
    "rayzor::gpu::GPUCompute", "add",          instance, "rayzor_gpu_compute_add",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor::gpu::GPUCompute", "sub",          instance, "rayzor_gpu_compute_sub",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor::gpu::GPUCompute", "mul",          instance, "rayzor_gpu_compute_mul",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor::gpu::GPUCompute", "div",          instance, "rayzor_gpu_compute_div",           [Ptr, Ptr, Ptr] => Ptr;
    // Unary elementwise ops: (self, a) -> result
    "rayzor::gpu::GPUCompute", "neg",          instance, "rayzor_gpu_compute_neg",           [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "abs",          instance, "rayzor_gpu_compute_abs",           [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "sqrt",         instance, "rayzor_gpu_compute_sqrt",          [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "exp",          instance, "rayzor_gpu_compute_exp",           [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "log",          instance, "rayzor_gpu_compute_log",           [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "relu",         instance, "rayzor_gpu_compute_relu",          [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "sigmoid",      instance, "rayzor_gpu_compute_sigmoid",       [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "tanh",         instance, "rayzor_gpu_compute_tanh",          [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "gelu",         instance, "rayzor_gpu_compute_gelu",          [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GPUCompute", "silu",         instance, "rayzor_gpu_compute_silu",          [Ptr, Ptr]      => Ptr;
    // Reductions: (self, buf) -> f64
    "rayzor::gpu::GPUCompute", "sum",          instance, "rayzor_gpu_compute_sum",           [Ptr, Ptr]      => F64;
    "rayzor::gpu::GPUCompute", "mean",         instance, "rayzor_gpu_compute_mean",          [Ptr, Ptr]      => F64;
    "rayzor::gpu::GPUCompute", "max",          instance, "rayzor_gpu_compute_max",           [Ptr, Ptr]      => F64;
    "rayzor::gpu::GPUCompute", "min",          instance, "rayzor_gpu_compute_min",           [Ptr, Ptr]      => F64;
    // Dot product: (self, a, b) -> f64
    "rayzor::gpu::GPUCompute", "dot",          instance, "rayzor_gpu_compute_dot",           [Ptr, Ptr, Ptr] => F64;
    // Matmul: (self, a, b, m, k, n) -> GpuBuffer
    "rayzor::gpu::GPUCompute", "matmul",       instance, "rayzor_gpu_compute_matmul",        [Ptr, Ptr, Ptr, I64, I64, I64] => Ptr;
    // Q4_K weights straight to the shader: no CPU dequant, ~7x less upload.
    "rayzor::gpu::GPUCompute", "matmulQ4K",    instance, "rayzor_gpu_compute_matmul_q4k",    [Ptr, Ptr, Ptr, I64, I64, I64] => Ptr;
    "rayzor::gpu::GPUCompute", "bufferFromBytes", instance, "rayzor_gpu_compute_buffer_from_bytes", [Ptr, I64, I64] => Ptr;
    "rayzor::gpu::GPUCompute", "writeBytes",    instance, "rayzor_gpu_compute_write_bytes",   [Ptr, Ptr, I64, I64] => Bool;
    // Batch matmul: (self, a, b, batch, m, k, n) -> GpuBuffer
    "rayzor::gpu::GPUCompute", "batchMatmul",  instance, "rayzor_gpu_compute_batch_matmul",  [Ptr, Ptr, Ptr, I64, I64, I64, I64] => Ptr;
    // Transformer primitives: (self, ...) -> GpuBuffer
    "rayzor::gpu::GPUCompute", "rmsNorm", instance, "rayzor_gpu_compute_rms_norm", [Ptr, Ptr, Ptr, I64, F64] => Ptr;
    "rayzor::gpu::GPUCompute", "rope",    instance, "rayzor_gpu_compute_rope",     [Ptr, Ptr, Ptr, Ptr, I64, I64, I64, I64, I64] => Ptr;
    // Structured buffer ops: (self, ...) -> result
    "rayzor::gpu::GPUCompute", "createStructBuffer", instance, "rayzor_gpu_compute_create_struct_buffer", [Ptr, Ptr, I64, I64] => Ptr;
    "rayzor::gpu::GPUCompute", "allocStructBuffer",  instance, "rayzor_gpu_compute_alloc_struct_buffer",  [Ptr, I64, I64]      => Ptr;
    "rayzor::gpu::GPUCompute", "readStructFloat",    instance, "rayzor_gpu_compute_read_struct_float",    [Ptr, Ptr, I64, I64, I64] => F64;
    "rayzor::gpu::GPUCompute", "readStructInt",      instance, "rayzor_gpu_compute_read_struct_int",      [Ptr, Ptr, I64, I64, I64] => I64;
    // GpuBuffer instance methods
    "rayzor::gpu::GpuBuffer",  "numel",        instance, "rayzor_gpu_compute_buffer_numel",  [Ptr]           => I64;
    "rayzor::gpu::GpuBuffer",  "dtype",        instance, "rayzor_gpu_compute_buffer_dtype",  [Ptr]           => I64;
    // GpuBuffer @:op overloads — `a + b` desugars to `a.add(b)` etc.
    // Reuses binary_lazy under the hood (no ctx needed; lazy DAG only).
    "rayzor::gpu::GpuBuffer",  "add",          instance, "rayzor_gpu_buffer_add",            [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GpuBuffer",  "sub",          instance, "rayzor_gpu_buffer_sub",            [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GpuBuffer",  "mul",          instance, "rayzor_gpu_buffer_mul",            [Ptr, Ptr]      => Ptr;
    "rayzor::gpu::GpuBuffer",  "div",          instance, "rayzor_gpu_buffer_div",            [Ptr, Ptr]      => Ptr;

    // ======================================================================
    // GPU Graphics (render pipeline)
    // ======================================================================

    // GPUDevice lifecycle
    "rayzor::gpu::GPUDevice", "create",       static,   "rayzor_gpu_gfx_device_create",      []              => Ptr;
    "rayzor::gpu::GPUDevice", "destroy",      instance, "rayzor_gpu_gfx_device_destroy",     [Ptr]           => Void;
    "rayzor::gpu::GPUDevice", "isAvailable",  static,   "rayzor_gpu_gfx_is_available",       []              => I64;

    // ShaderModule (Haxe-friendly: accepts HaxeString* directly)
    "rayzor::gpu::ShaderModule", "create",    static,   "rayzor_gpu_gfx_shader_create_hx",   [Ptr, Ptr, Ptr, Ptr] => Ptr;
    "rayzor::gpu::ShaderModule", "destroy",   instance, "rayzor_gpu_gfx_shader_destroy",     [Ptr]           => Void;

    // Buffer (graphics)
    "rayzor::gpu::GfxBuffer", "create",           static,   "rayzor_gpu_gfx_buffer_create",           [Ptr, I64, I64] => Ptr;
    "rayzor::gpu::GfxBuffer", "createWithData",   static,   "rayzor_gpu_gfx_buffer_create_with_data", [Ptr, Ptr, I64, I64] => Ptr;
    "rayzor::gpu::GfxBuffer", "write",            instance, "rayzor_gpu_gfx_buffer_write",            [Ptr, Ptr, I64, Ptr, I64] => Void;
    "rayzor::gpu::GfxBuffer", "destroy",          instance, "rayzor_gpu_gfx_buffer_destroy",          [Ptr]           => Void;

    // Texture
    "rayzor::gpu::Texture", "create",         static,   "rayzor_gpu_gfx_texture_create",     [Ptr, I64, I64, I64, I64] => Ptr;
    "rayzor::gpu::Texture", "write",          instance, "rayzor_gpu_gfx_texture_write",      [Ptr, Ptr, Ptr, I64, I64] => Void;
    "rayzor::gpu::Texture", "getView",        instance, "rayzor_gpu_gfx_texture_get_view",   [Ptr]           => Ptr;
    "rayzor::gpu::Texture", "destroy",        instance, "rayzor_gpu_gfx_texture_destroy",    [Ptr]           => Void;
    "rayzor::gpu::Texture", "readPixels",     instance, "rayzor_gpu_gfx_texture_read_pixels", [Ptr, Ptr, Ptr, I64] => I64;

    // Sampler
    "rayzor::gpu::Sampler", "create",         static,   "rayzor_gpu_gfx_sampler_create",     [Ptr, I64, I64, I64] => Ptr;
    "rayzor::gpu::Sampler", "destroy",        instance, "rayzor_gpu_gfx_sampler_destroy",    [Ptr]           => Void;

    // RenderPipeline builder
    "rayzor::gpu::RenderPipeline", "begin",   static,   "rayzor_gpu_gfx_pipeline_begin",     []              => Ptr;
    "rayzor::gpu::RenderPipeline", "setShader", instance, "rayzor_gpu_gfx_pipeline_set_shader", [Ptr, Ptr]   => Void;
    "rayzor::gpu::RenderPipeline", "setFormat", instance, "rayzor_gpu_gfx_pipeline_set_format", [Ptr, I64]   => Void;
    "rayzor::gpu::RenderPipeline", "setTopology", instance, "rayzor_gpu_gfx_pipeline_set_topology", [Ptr, I64] => Void;
    "rayzor::gpu::RenderPipeline", "setCull",   instance, "rayzor_gpu_gfx_pipeline_set_cull",   [Ptr, I64]   => Void;
    "rayzor::gpu::RenderPipeline", "build",     instance, "rayzor_gpu_gfx_pipeline_build",     [Ptr, Ptr]    => Ptr;
    "rayzor::gpu::RenderPipeline", "destroy",   instance, "rayzor_gpu_gfx_pipeline_destroy",   [Ptr]         => Void;

    // BindGroup
    "rayzor::gpu::BindGroupLayout", "create", static,   "rayzor_gpu_gfx_bind_group_layout_create", [Ptr, I64, Ptr, Ptr] => Ptr;
    "rayzor::gpu::BindGroupLayout", "destroy", instance, "rayzor_gpu_gfx_bind_group_layout_destroy", [Ptr]   => Void;
    "rayzor::gpu::BindGroup", "destroy",      instance, "rayzor_gpu_gfx_bind_group_destroy",  [Ptr]           => Void;

    // Render (Haxe-friendly simplified APIs)
    "rayzor::gpu::Renderer", "renderTriangles", static, "rayzor_gpu_gfx_render_triangles", [Ptr, Ptr, Ptr, I64, F64, F64, F64, F64] => Void;
    "rayzor::gpu::Texture", "toBytes",          instance, "rayzor_gpu_gfx_texture_to_bytes", [Ptr, Ptr] => Ptr;

    // Surface
    "rayzor::gpu::Surface", "create",         static,   "rayzor_gpu_gfx_surface_create",     [Ptr, Ptr, Ptr, I64, I64] => Ptr;
    // Surface.createCanvas is available on all platforms — on WASM the host provides it,
    // on native it's a no-op (use Surface.create with raw window handles instead).
    "rayzor::gpu::Surface", "createCanvas",   static,   "rayzor_gpu_gfx_surface_create_canvas", [Ptr, Ptr, I64, I64] => Ptr;
    "rayzor::gpu::Surface", "getTexture",     instance, "rayzor_gpu_gfx_surface_get_texture",[Ptr]           => Ptr;
    "rayzor::gpu::Surface", "present",        instance, "rayzor_gpu_gfx_surface_present",    [Ptr]           => Void;
    "rayzor::gpu::Surface", "resize",         instance, "rayzor_gpu_gfx_surface_resize",     [Ptr, Ptr, I64, I64] => Void;
    "rayzor::gpu::Surface", "getFormat",      instance, "rayzor_gpu_gfx_surface_get_format", [Ptr]           => I64;
    "rayzor::gpu::Surface", "destroy",        instance, "rayzor_gpu_gfx_surface_destroy",    [Ptr]           => Void;

    // CommandEncoder (multi-pass)
    "rayzor::gpu::CommandEncoder", "create",          static,   "rayzor_gpu_gfx_cmd_create",           []                  => Ptr;
    "rayzor::gpu::CommandEncoder", "beginPass",       instance, "rayzor_gpu_gfx_cmd_begin_pass",       [Ptr, Ptr, I64, F64, F64, F64, F64, Ptr] => Void;
    "rayzor::gpu::CommandEncoder", "endPass",         instance, "rayzor_gpu_gfx_cmd_end_pass",         [Ptr]               => Void;
    "rayzor::gpu::CommandEncoder", "submit",          instance, "rayzor_gpu_gfx_cmd_submit",           [Ptr, Ptr]          => Void;
    "rayzor::gpu::CommandEncoder", "setPipeline",     instance, "rayzor_gpu_gfx_cmd_set_pipeline",     [Ptr, Ptr]          => Void;
    "rayzor::gpu::CommandEncoder", "draw",            instance, "rayzor_gpu_gfx_cmd_draw",             [Ptr, I64, I64, I64, I64] => Void;
    "rayzor::gpu::CommandEncoder", "drawIndexed",     instance, "rayzor_gpu_gfx_cmd_draw_indexed",     [Ptr, I64, I64, I64, I64, I64] => Void;
    "rayzor::gpu::CommandEncoder", "setVertexBuffer", instance, "rayzor_gpu_gfx_cmd_set_vertex_buffer",[Ptr, I64, Ptr]     => Void;
    "rayzor::gpu::CommandEncoder", "setIndexBuffer",  instance, "rayzor_gpu_gfx_cmd_set_index_buffer", [Ptr, Ptr, I64]     => Void;
    "rayzor::gpu::CommandEncoder", "setBindGroup",    instance, "rayzor_gpu_gfx_cmd_set_bind_group",   [Ptr, I64, Ptr]     => Void;
    "rayzor::gpu::CommandEncoder", "setViewport",     instance, "rayzor_gpu_gfx_cmd_set_viewport",     [Ptr, F64, F64, F64, F64, F64, F64] => Void;
    "rayzor::gpu::CommandEncoder", "setScissor",      instance, "rayzor_gpu_gfx_cmd_set_scissor",      [Ptr, I64, I64, I64, I64]           => Void;
    "rayzor::gpu::CommandEncoder", "beginPassMRT",   instance, "rayzor_gpu_gfx_cmd_begin_pass_mrt",  [Ptr, I64, Ptr, Ptr, Ptr, Ptr]      => Void;

    // Pipeline MRT
    "rayzor::gpu::RenderPipeline", "addColorTarget", instance, "rayzor_gpu_gfx_pipeline_add_color_target", [Ptr, I64] => Void;
}

// ============================================================================
// Plugin exports (called by host via dlopen/dlsym) — native only
// ============================================================================
#[cfg(feature = "native")]
mod native_plugin {
    use super::*;
    use std::ffi::c_void;
    use std::sync::LazyLock;

    fn copy_method(method: &NativeMethodDesc) -> NativeMethodDesc {
        NativeMethodDesc {
            symbol_name: method.symbol_name,
            symbol_name_len: method.symbol_name_len,
            class_name: method.class_name,
            class_name_len: method.class_name_len,
            method_name: method.method_name,
            method_name_len: method.method_name_len,
            is_static: method.is_static,
            param_count: method.param_count,
            return_type: method.return_type,
            param_types: method.param_types,
        }
    }

    static ALL_METHODS: LazyLock<Vec<NativeMethodDesc>> = LazyLock::new(|| {
        let mut methods: Vec<_> = GPU_METHODS.iter().map(copy_method).collect();
        #[cfg(feature = "webgpu-backend")]
        methods.extend(XGPU_METHODS.iter().map(copy_method));
        methods
    });

    /// Symbol table entry for plugin registration
    #[repr(C)]
    pub struct SymbolEntry {
        pub name: *const u8,
        pub name_len: usize,
        pub ptr: *const c_void,
    }

    /// Plugin initialization — returns a flat symbol table for JIT linking.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rayzor_gpu_plugin_init(out_count: *mut usize) -> *const SymbolEntry {
        let symbols = collect_symbols();
        let count = symbols.len();
        let ptr = symbols.as_ptr();
        std::mem::forget(symbols); // caller does not free — lives for process lifetime
        if !out_count.is_null() {
            unsafe {
                *out_count = count;
            }
        }
        ptr
    }

    /// Returns method descriptors for compiler-side registration.
    ///
    /// The compiler reads these to auto-generate method mappings and extern
    /// declarations — no manual MIR wrappers or compiler core changes needed.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn rayzor_gpu_plugin_describe(
        out_count: *mut usize,
    ) -> *const NativeMethodDesc {
        if !out_count.is_null() {
            unsafe {
                *out_count = ALL_METHODS.len();
            }
        }
        ALL_METHODS.as_ptr()
    }

    /// Rust-callable API returning runtime symbols.
    #[allow(unused_mut)]
    pub fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
        let mut symbols = vec![
            // Device lifecycle
            (
                "rayzor_gpu_compute_matmul_q4k",
                ops::rayzor_gpu_compute_matmul_q4k as *const u8,
            ),
            (
                "rayzor_gpu_compute_buffer_from_bytes",
                ops::rayzor_gpu_compute_buffer_from_bytes as *const u8,
            ),
            (
                "rayzor_gpu_compute_write_bytes",
                ops::rayzor_gpu_compute_write_bytes as *const u8,
            ),
            (
                "rayzor_gpu_compute_create",
                device::rayzor_gpu_compute_create as *const u8,
            ),
            #[cfg(feature = "webgpu-backend")]
            (
                "rayzor_gpu_compute_from_device",
                device::rayzor_gpu_compute_from_device as *const u8,
            ),
            (
                "rayzor_gpu_compute_destroy",
                device::rayzor_gpu_compute_destroy as *const u8,
            ),
            (
                "rayzor_gpu_compute_is_available",
                device::rayzor_gpu_compute_is_available as *const u8,
            ),
            // Buffer management
            (
                "rayzor_gpu_compute_create_buffer",
                buffer::rayzor_gpu_compute_create_buffer as *const u8,
            ),
            (
                "rayzor_gpu_compute_alloc_buffer",
                buffer::rayzor_gpu_compute_alloc_buffer as *const u8,
            ),
            (
                "rayzor_gpu_compute_to_tensor",
                buffer::rayzor_gpu_compute_to_tensor as *const u8,
            ),
            (
                "rayzor_gpu_compute_free_buffer",
                buffer::rayzor_gpu_compute_free_buffer as *const u8,
            ),
            (
                "rayzor_gpu_compute_buffer_numel",
                buffer::rayzor_gpu_compute_buffer_numel as *const u8,
            ),
            (
                "rayzor_gpu_compute_buffer_dtype",
                buffer::rayzor_gpu_compute_buffer_dtype as *const u8,
            ),
            // Binary elementwise ops
            (
                "rayzor_gpu_compute_add",
                ops::rayzor_gpu_compute_add as *const u8,
            ),
            (
                "rayzor_gpu_compute_sub",
                ops::rayzor_gpu_compute_sub as *const u8,
            ),
            (
                "rayzor_gpu_compute_mul",
                ops::rayzor_gpu_compute_mul as *const u8,
            ),
            (
                "rayzor_gpu_compute_div",
                ops::rayzor_gpu_compute_div as *const u8,
            ),
            // GpuBuffer @:op binary ops — same dispatch, no ctx parameter
            (
                "rayzor_gpu_buffer_add",
                ops::rayzor_gpu_buffer_add as *const u8,
            ),
            (
                "rayzor_gpu_buffer_sub",
                ops::rayzor_gpu_buffer_sub as *const u8,
            ),
            (
                "rayzor_gpu_buffer_mul",
                ops::rayzor_gpu_buffer_mul as *const u8,
            ),
            (
                "rayzor_gpu_buffer_div",
                ops::rayzor_gpu_buffer_div as *const u8,
            ),
            // Unary elementwise ops
            (
                "rayzor_gpu_compute_neg",
                ops::rayzor_gpu_compute_neg as *const u8,
            ),
            (
                "rayzor_gpu_compute_abs",
                ops::rayzor_gpu_compute_abs as *const u8,
            ),
            (
                "rayzor_gpu_compute_sqrt",
                ops::rayzor_gpu_compute_sqrt as *const u8,
            ),
            (
                "rayzor_gpu_compute_exp",
                ops::rayzor_gpu_compute_exp as *const u8,
            ),
            (
                "rayzor_gpu_compute_log",
                ops::rayzor_gpu_compute_log as *const u8,
            ),
            (
                "rayzor_gpu_compute_relu",
                ops::rayzor_gpu_compute_relu as *const u8,
            ),
            (
                "rayzor_gpu_compute_sigmoid",
                ops::rayzor_gpu_compute_sigmoid as *const u8,
            ),
            (
                "rayzor_gpu_compute_tanh",
                ops::rayzor_gpu_compute_tanh as *const u8,
            ),
            (
                "rayzor_gpu_compute_gelu",
                ops::rayzor_gpu_compute_gelu as *const u8,
            ),
            (
                "rayzor_gpu_compute_silu",
                ops::rayzor_gpu_compute_silu as *const u8,
            ),
            // Reductions
            (
                "rayzor_gpu_compute_sum",
                ops::rayzor_gpu_compute_sum as *const u8,
            ),
            (
                "rayzor_gpu_compute_mean",
                ops::rayzor_gpu_compute_mean as *const u8,
            ),
            (
                "rayzor_gpu_compute_max",
                ops::rayzor_gpu_compute_max as *const u8,
            ),
            (
                "rayzor_gpu_compute_min",
                ops::rayzor_gpu_compute_min as *const u8,
            ),
            // Dot product
            (
                "rayzor_gpu_compute_dot",
                ops::rayzor_gpu_compute_dot as *const u8,
            ),
            // Matmul
            (
                "rayzor_gpu_compute_matmul",
                ops::rayzor_gpu_compute_matmul as *const u8,
            ),
            (
                "rayzor_gpu_compute_batch_matmul",
                ops::rayzor_gpu_compute_batch_matmul as *const u8,
            ),
            // Transformer primitives
            (
                "rayzor_gpu_compute_rms_norm",
                ops::rayzor_gpu_compute_rms_norm as *const u8,
            ),
            (
                "rayzor_gpu_compute_rope",
                ops::rayzor_gpu_compute_rope as *const u8,
            ),
            // Structured buffer ops
            (
                "rayzor_gpu_compute_create_struct_buffer",
                buffer::rayzor_gpu_compute_create_struct_buffer as *const u8,
            ),
            (
                "rayzor_gpu_compute_alloc_struct_buffer",
                buffer::rayzor_gpu_compute_alloc_struct_buffer as *const u8,
            ),
            (
                "rayzor_gpu_compute_read_struct_float",
                buffer::rayzor_gpu_compute_read_struct_float as *const u8,
            ),
            (
                "rayzor_gpu_compute_read_struct_int",
                buffer::rayzor_gpu_compute_read_struct_int as *const u8,
            ),
        ];

        // Graphics rendering symbols (only when webgpu-backend is compiled)
        #[cfg(feature = "webgpu-backend")]
        {
            let gfx_symbols: Vec<(&'static str, *const u8)> = vec![
                (
                    "rayzor_gpu_gfx_device_create",
                    graphics::rayzor_gpu_gfx_device_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_device_destroy",
                    graphics::rayzor_gpu_gfx_device_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_is_available",
                    graphics::rayzor_gpu_gfx_is_available as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_create",
                    graphics::rayzor_gpu_gfx_buffer_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_create_with_data",
                    graphics::rayzor_gpu_gfx_buffer_create_with_data as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_write",
                    graphics::rayzor_gpu_gfx_buffer_write as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_destroy",
                    graphics::rayzor_gpu_gfx_buffer_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_shader_create",
                    graphics::shader::rayzor_gpu_gfx_shader_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_shader_create_hx",
                    graphics::shader::rayzor_gpu_gfx_shader_create_hx as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_shader_destroy",
                    graphics::shader::rayzor_gpu_gfx_shader_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_begin",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_begin as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_shader",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_set_shader as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_format",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_set_format as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_topology",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_set_topology as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_cull",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_set_cull as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_build",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_build as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_destroy",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_create",
                    graphics::texture::rayzor_gpu_gfx_texture_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_get_view",
                    graphics::texture::rayzor_gpu_gfx_texture_get_view as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_read_pixels",
                    graphics::texture::rayzor_gpu_gfx_texture_read_pixels as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_destroy",
                    graphics::texture::rayzor_gpu_gfx_texture_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_render_submit",
                    graphics::render_pass::rayzor_gpu_gfx_render_submit as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_render_triangles",
                    graphics::render_pass::rayzor_gpu_gfx_render_triangles as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_to_bytes",
                    graphics::render_pass::rayzor_gpu_gfx_texture_to_bytes as *const u8,
                ),
                // Command encoder (multi-pass)
                (
                    "rayzor_gpu_gfx_cmd_create",
                    graphics::command::rayzor_gpu_gfx_cmd_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_begin_pass",
                    graphics::command::rayzor_gpu_gfx_cmd_begin_pass as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_pipeline",
                    graphics::command::rayzor_gpu_gfx_cmd_set_pipeline as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_vertex_buffer",
                    graphics::command::rayzor_gpu_gfx_cmd_set_vertex_buffer as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_index_buffer",
                    graphics::command::rayzor_gpu_gfx_cmd_set_index_buffer as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_bind_group",
                    graphics::command::rayzor_gpu_gfx_cmd_set_bind_group as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_draw",
                    graphics::command::rayzor_gpu_gfx_cmd_draw as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_draw_indexed",
                    graphics::command::rayzor_gpu_gfx_cmd_draw_indexed as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_viewport",
                    graphics::command::rayzor_gpu_gfx_cmd_set_viewport as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_set_scissor",
                    graphics::command::rayzor_gpu_gfx_cmd_set_scissor as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_end_pass",
                    graphics::command::rayzor_gpu_gfx_cmd_end_pass as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_submit",
                    graphics::command::rayzor_gpu_gfx_cmd_submit as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_cmd_begin_pass_mrt",
                    graphics::command::rayzor_gpu_gfx_cmd_begin_pass_mrt as *const u8,
                ),
                // Pipeline MRT
                (
                    "rayzor_gpu_gfx_pipeline_add_color_target",
                    graphics::pipeline::rayzor_gpu_gfx_pipeline_add_color_target as *const u8,
                ),
                // Haxe-friendly wrappers
                (
                    "rayzor_gpu_gfx_render_with_vb",
                    graphics::haxe_api::rayzor_gpu_gfx_render_with_vb as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_render_indexed",
                    graphics::haxe_api::rayzor_gpu_gfx_render_indexed as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_render_with_depth",
                    graphics::haxe_api::rayzor_gpu_gfx_render_with_depth as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_render_with_bindings",
                    graphics::haxe_api::rayzor_gpu_gfx_render_with_bindings as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_from_bytes",
                    graphics::haxe_api::rayzor_gpu_gfx_buffer_from_bytes as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_buffer_write_bytes",
                    graphics::haxe_api::rayzor_gpu_gfx_buffer_write_bytes as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_bind_group_layout_uniform",
                    graphics::haxe_api::rayzor_gpu_gfx_bind_group_layout_uniform as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_bind_group_single",
                    graphics::haxe_api::rayzor_gpu_gfx_bind_group_single as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_vertex_layout_simple",
                    graphics::haxe_api::rayzor_gpu_gfx_pipeline_set_vertex_layout_simple
                        as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_set_depth_simple",
                    graphics::haxe_api::rayzor_gpu_gfx_pipeline_set_depth_simple as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_pipeline_add_layout",
                    graphics::haxe_api::rayzor_gpu_gfx_pipeline_add_layout as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_depth_texture_create",
                    graphics::haxe_api::rayzor_gpu_gfx_depth_texture_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_sampler_linear",
                    graphics::haxe_api::rayzor_gpu_gfx_sampler_linear as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_texture_upload",
                    graphics::haxe_api::rayzor_gpu_gfx_texture_upload as *const u8,
                ),
                // Surface
                (
                    "rayzor_gpu_gfx_surface_create",
                    graphics::surface::rayzor_gpu_gfx_surface_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_surface_get_texture",
                    graphics::surface::rayzor_gpu_gfx_surface_get_texture as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_surface_present",
                    graphics::surface::rayzor_gpu_gfx_surface_present as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_surface_resize",
                    graphics::surface::rayzor_gpu_gfx_surface_resize as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_surface_get_format",
                    graphics::surface::rayzor_gpu_gfx_surface_get_format as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_surface_destroy",
                    graphics::surface::rayzor_gpu_gfx_surface_destroy as *const u8,
                ),
                // Texture write
                (
                    "rayzor_gpu_gfx_texture_write",
                    graphics::texture::rayzor_gpu_gfx_texture_write as *const u8,
                ),
                // Sampler
                (
                    "rayzor_gpu_gfx_sampler_create",
                    graphics::texture::rayzor_gpu_gfx_sampler_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_sampler_destroy",
                    graphics::texture::rayzor_gpu_gfx_sampler_destroy as *const u8,
                ),
                // BindGroup
                (
                    "rayzor_gpu_gfx_bind_group_layout_create",
                    graphics::bind_group::rayzor_gpu_gfx_bind_group_layout_create as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_bind_group_layout_destroy",
                    graphics::bind_group::rayzor_gpu_gfx_bind_group_layout_destroy as *const u8,
                ),
                (
                    "rayzor_gpu_gfx_bind_group_destroy",
                    graphics::bind_group::rayzor_gpu_gfx_bind_group_destroy as *const u8,
                ),
            ];
            symbols.extend(gfx_symbols);
        }

        #[cfg(feature = "webgpu-backend")]
        symbols.extend(xgpu_runtime_symbols());

        symbols
    }

    fn collect_symbols() -> Vec<SymbolEntry> {
        get_runtime_symbols()
            .into_iter()
            .map(|(name, ptr)| SymbolEntry {
                name: name.as_ptr(),
                name_len: name.len(),
                ptr: ptr as *const c_void,
            })
            .collect()
    }

    // Universal rpkg entry point — single export for both symbols and method descriptors
    rayzor_plugin::rpkg_entry!(ALL_METHODS, get_runtime_symbols);

    /// GPU compute plugin implementing RuntimePlugin trait
    pub struct GpuComputePlugin;

    impl rayzor_plugin::RuntimePlugin for GpuComputePlugin {
        fn name(&self) -> &str {
            "rayzor_gpu_compute"
        }

        fn runtime_symbols(&self) -> Vec<(&'static str, *const u8)> {
            get_runtime_symbols()
        }
    }
} // mod native_plugin
#[cfg(feature = "native")]
pub use native_plugin::*;

#[cfg(all(test, feature = "native", feature = "webgpu-backend"))]
mod xgpu_adapter_tests {
    use std::collections::HashSet;

    #[test]
    fn portable_and_rayzor_extensions_share_one_method_catalog() {
        let mut count = 0;
        let pointer = unsafe { super::rayzor_gpu_plugin_describe(&mut count) };
        let methods = unsafe { std::slice::from_raw_parts(pointer, count) };
        let mut keys = HashSet::new();
        let mut expected = HashSet::from([
            "rayzor.gpu.GpuBuffer.destroy".to_owned(),
            "rayzor.gpu.GpuBuffer.numel".to_owned(),
            "rayzor.gpu.GPUCompute.fromDevice".to_owned(),
        ]);
        for method in methods {
            let class = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    method.class_name,
                    method.class_name_len,
                ))
            }
            .replace("::", ".");
            let name = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    method.method_name,
                    method.method_name_len,
                ))
            };
            let key = format!("{class}.{name}");
            assert!(keys.insert(key.clone()), "duplicate GPU method {key}");
            expected.remove(&key);
        }
        assert!(
            expected.is_empty(),
            "missing shared GPU methods: {expected:?}"
        );
    }
}
