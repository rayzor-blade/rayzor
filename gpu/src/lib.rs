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

// All extern "C" functions in this crate are FFI entry points called by the JIT runtime.
#![allow(clippy::missing_safety_doc)]

pub mod buffer;
pub mod codegen;
pub mod device;
pub mod kernel_cache;
pub mod kernel_ir;
pub mod lazy;
pub mod ops;

pub mod backend;

#[cfg(feature = "metal-backend")]
pub mod metal;

#[cfg(feature = "webgpu-backend")]
pub mod wgpu_backend;

#[cfg(feature = "webgpu-backend")]
pub mod graphics;

#[cfg(feature = "cuda-backend")]
pub mod cuda;

use rayzor_plugin::{declare_native_methods, NativeMethodDesc};
use std::ffi::c_void;

// ============================================================================
// Method descriptor table (read by compiler at plugin load time)
// ============================================================================

declare_native_methods! {
    GPU_METHODS;
    // GPUCompute lifecycle (static)
    "rayzor_gpu_GPUCompute", "create",       static,   "rayzor_gpu_compute_create",        []              => Ptr;
    "rayzor_gpu_GPUCompute", "isAvailable",  static,   "rayzor_gpu_compute_is_available",  []              => Bool;
    // GPUCompute instance methods (self = Ptr is first param)
    "rayzor_gpu_GPUCompute", "destroy",      instance, "rayzor_gpu_compute_destroy",       [Ptr]           => Void;
    "rayzor_gpu_GPUCompute", "createBuffer", instance, "rayzor_gpu_compute_create_buffer", [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "allocBuffer",  instance, "rayzor_gpu_compute_alloc_buffer",  [Ptr, I64, I64] => Ptr;
    "rayzor_gpu_GPUCompute", "toTensor",     instance, "rayzor_gpu_compute_to_tensor",     [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "freeBuffer",   instance, "rayzor_gpu_compute_free_buffer",   [Ptr, Ptr]      => Void;
    // Binary elementwise ops: (self, a, b) -> result
    "rayzor_gpu_GPUCompute", "add",          instance, "rayzor_gpu_compute_add",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor_gpu_GPUCompute", "sub",          instance, "rayzor_gpu_compute_sub",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor_gpu_GPUCompute", "mul",          instance, "rayzor_gpu_compute_mul",           [Ptr, Ptr, Ptr] => Ptr;
    "rayzor_gpu_GPUCompute", "div",          instance, "rayzor_gpu_compute_div",           [Ptr, Ptr, Ptr] => Ptr;
    // Unary elementwise ops: (self, a) -> result
    "rayzor_gpu_GPUCompute", "neg",          instance, "rayzor_gpu_compute_neg",           [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "abs",          instance, "rayzor_gpu_compute_abs",           [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "sqrt",         instance, "rayzor_gpu_compute_sqrt",          [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "exp",          instance, "rayzor_gpu_compute_exp",           [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "log",          instance, "rayzor_gpu_compute_log",           [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "relu",         instance, "rayzor_gpu_compute_relu",          [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "sigmoid",      instance, "rayzor_gpu_compute_sigmoid",       [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "tanh",         instance, "rayzor_gpu_compute_tanh",          [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "gelu",         instance, "rayzor_gpu_compute_gelu",          [Ptr, Ptr]      => Ptr;
    "rayzor_gpu_GPUCompute", "silu",         instance, "rayzor_gpu_compute_silu",          [Ptr, Ptr]      => Ptr;
    // Reductions: (self, buf) -> f64
    "rayzor_gpu_GPUCompute", "sum",          instance, "rayzor_gpu_compute_sum",           [Ptr, Ptr]      => F64;
    "rayzor_gpu_GPUCompute", "mean",         instance, "rayzor_gpu_compute_mean",          [Ptr, Ptr]      => F64;
    "rayzor_gpu_GPUCompute", "max",          instance, "rayzor_gpu_compute_max",           [Ptr, Ptr]      => F64;
    "rayzor_gpu_GPUCompute", "min",          instance, "rayzor_gpu_compute_min",           [Ptr, Ptr]      => F64;
    // Dot product: (self, a, b) -> f64
    "rayzor_gpu_GPUCompute", "dot",          instance, "rayzor_gpu_compute_dot",           [Ptr, Ptr, Ptr] => F64;
    // Matmul: (self, a, b, m, k, n) -> GpuBuffer
    "rayzor_gpu_GPUCompute", "matmul",       instance, "rayzor_gpu_compute_matmul",        [Ptr, Ptr, Ptr, I64, I64, I64] => Ptr;
    // Batch matmul: (self, a, b, batch, m, k, n) -> GpuBuffer
    "rayzor_gpu_GPUCompute", "batchMatmul",  instance, "rayzor_gpu_compute_batch_matmul",  [Ptr, Ptr, Ptr, I64, I64, I64, I64] => Ptr;
    // Structured buffer ops: (self, ...) -> result
    "rayzor_gpu_GPUCompute", "createStructBuffer", instance, "rayzor_gpu_compute_create_struct_buffer", [Ptr, Ptr, I64, I64] => Ptr;
    "rayzor_gpu_GPUCompute", "allocStructBuffer",  instance, "rayzor_gpu_compute_alloc_struct_buffer",  [Ptr, I64, I64]      => Ptr;
    "rayzor_gpu_GPUCompute", "readStructFloat",    instance, "rayzor_gpu_compute_read_struct_float",    [Ptr, Ptr, I64, I64, I64] => F64;
    "rayzor_gpu_GPUCompute", "readStructInt",      instance, "rayzor_gpu_compute_read_struct_int",      [Ptr, Ptr, I64, I64, I64] => I64;
    // GpuBuffer instance methods
    "rayzor_gpu_GpuBuffer",  "numel",        instance, "rayzor_gpu_compute_buffer_numel",  [Ptr]           => I64;
    "rayzor_gpu_GpuBuffer",  "dtype",        instance, "rayzor_gpu_compute_buffer_dtype",  [Ptr]           => I64;

    // ======================================================================
    // GPU Graphics (render pipeline)
    // ======================================================================

    // GPUDevice lifecycle
    "rayzor_gpu_GPUDevice", "create",       static,   "rayzor_gpu_gfx_device_create",      []              => Ptr;
    "rayzor_gpu_GPUDevice", "destroy",      instance, "rayzor_gpu_gfx_device_destroy",     [Ptr]           => Void;
    "rayzor_gpu_GPUDevice", "isAvailable",  static,   "rayzor_gpu_gfx_is_available",       []              => I64;

    // ShaderModule (Haxe-friendly: accepts HaxeString* directly)
    "rayzor_gpu_ShaderModule", "create",    static,   "rayzor_gpu_gfx_shader_create_hx",   [Ptr, Ptr, Ptr, Ptr] => Ptr;
    "rayzor_gpu_ShaderModule", "destroy",   instance, "rayzor_gpu_gfx_shader_destroy",     [Ptr]           => Void;

    // Buffer (graphics)
    "rayzor_gpu_GfxBuffer", "create",           static,   "rayzor_gpu_gfx_buffer_create",           [Ptr, I64, I64] => Ptr;
    "rayzor_gpu_GfxBuffer", "createWithData",   static,   "rayzor_gpu_gfx_buffer_create_with_data", [Ptr, Ptr, I64, I64] => Ptr;
    "rayzor_gpu_GfxBuffer", "write",            instance, "rayzor_gpu_gfx_buffer_write",            [Ptr, Ptr, I64, Ptr, I64] => Void;
    "rayzor_gpu_GfxBuffer", "destroy",          instance, "rayzor_gpu_gfx_buffer_destroy",          [Ptr]           => Void;

    // Texture
    "rayzor_gpu_Texture", "create",         static,   "rayzor_gpu_gfx_texture_create",     [Ptr, I64, I64, I64, I64] => Ptr;
    "rayzor_gpu_Texture", "write",          instance, "rayzor_gpu_gfx_texture_write",      [Ptr, Ptr, Ptr, I64, I64] => Void;
    "rayzor_gpu_Texture", "getView",        instance, "rayzor_gpu_gfx_texture_get_view",   [Ptr]           => Ptr;
    "rayzor_gpu_Texture", "destroy",        instance, "rayzor_gpu_gfx_texture_destroy",    [Ptr]           => Void;
    "rayzor_gpu_Texture", "readPixels",     instance, "rayzor_gpu_gfx_texture_read_pixels", [Ptr, Ptr, Ptr, I64] => I64;

    // Sampler
    "rayzor_gpu_Sampler", "create",         static,   "rayzor_gpu_gfx_sampler_create",     [Ptr, I64, I64, I64] => Ptr;
    "rayzor_gpu_Sampler", "destroy",        instance, "rayzor_gpu_gfx_sampler_destroy",    [Ptr]           => Void;

    // RenderPipeline builder
    "rayzor_gpu_RenderPipeline", "begin",   static,   "rayzor_gpu_gfx_pipeline_begin",     []              => Ptr;
    "rayzor_gpu_RenderPipeline", "setShader", instance, "rayzor_gpu_gfx_pipeline_set_shader", [Ptr, Ptr]   => Void;
    "rayzor_gpu_RenderPipeline", "setFormat", instance, "rayzor_gpu_gfx_pipeline_set_format", [Ptr, I64]   => Void;
    "rayzor_gpu_RenderPipeline", "setTopology", instance, "rayzor_gpu_gfx_pipeline_set_topology", [Ptr, I64] => Void;
    "rayzor_gpu_RenderPipeline", "setCull",   instance, "rayzor_gpu_gfx_pipeline_set_cull",   [Ptr, I64]   => Void;
    "rayzor_gpu_RenderPipeline", "build",     instance, "rayzor_gpu_gfx_pipeline_build",     [Ptr, Ptr]    => Ptr;
    "rayzor_gpu_RenderPipeline", "destroy",   instance, "rayzor_gpu_gfx_pipeline_destroy",   [Ptr]         => Void;

    // BindGroup
    "rayzor_gpu_BindGroupLayout", "create", static,   "rayzor_gpu_gfx_bind_group_layout_create", [Ptr, I64, Ptr, Ptr] => Ptr;
    "rayzor_gpu_BindGroupLayout", "destroy", instance, "rayzor_gpu_gfx_bind_group_layout_destroy", [Ptr]   => Void;
    "rayzor_gpu_BindGroup", "destroy",      instance, "rayzor_gpu_gfx_bind_group_destroy",  [Ptr]           => Void;

    // Render (Haxe-friendly simplified APIs)
    "rayzor_gpu_Renderer", "renderTriangles", static, "rayzor_gpu_gfx_render_triangles", [Ptr, Ptr, Ptr, I64, F64, F64, F64, F64] => Void;
    "rayzor_gpu_Texture", "toBytes",          instance, "rayzor_gpu_gfx_texture_to_bytes", [Ptr, Ptr] => Ptr;

    // Surface
    "rayzor_gpu_Surface", "present",        instance, "rayzor_gpu_gfx_surface_present",    [Ptr]           => Void;
    "rayzor_gpu_Surface", "destroy",        instance, "rayzor_gpu_gfx_surface_destroy",    [Ptr]           => Void;

    // CommandEncoder (multi-pass)
    "rayzor_gpu_CommandEncoder", "create",          static,   "rayzor_gpu_gfx_cmd_create",           []                  => Ptr;
    "rayzor_gpu_CommandEncoder", "endPass",         instance, "rayzor_gpu_gfx_cmd_end_pass",         [Ptr]               => Void;
    "rayzor_gpu_CommandEncoder", "submit",          instance, "rayzor_gpu_gfx_cmd_submit",           [Ptr, Ptr]          => Void;
    "rayzor_gpu_CommandEncoder", "setPipeline",     instance, "rayzor_gpu_gfx_cmd_set_pipeline",     [Ptr, Ptr]          => Void;
    "rayzor_gpu_CommandEncoder", "setVertexBuffer", instance, "rayzor_gpu_gfx_cmd_set_vertex_buffer",[Ptr, I64, Ptr]     => Void;
    "rayzor_gpu_CommandEncoder", "setIndexBuffer",  instance, "rayzor_gpu_gfx_cmd_set_index_buffer", [Ptr, Ptr, I64]     => Void;
    "rayzor_gpu_CommandEncoder", "setBindGroup",    instance, "rayzor_gpu_gfx_cmd_set_bind_group",   [Ptr, I64, Ptr]     => Void;
    "rayzor_gpu_CommandEncoder", "setViewport",     instance, "rayzor_gpu_gfx_cmd_set_viewport",     [Ptr, F64, F64, F64, F64, F64, F64] => Void;
    "rayzor_gpu_CommandEncoder", "setScissor",      instance, "rayzor_gpu_gfx_cmd_set_scissor",      [Ptr, I64, I64, I64, I64]           => Void;
}

// ============================================================================
// Plugin exports (called by host via dlopen/dlsym)
// ============================================================================

/// Symbol table entry for plugin registration
#[repr(C)]
pub struct SymbolEntry {
    pub name: *const u8,
    pub name_len: usize,
    pub ptr: *const c_void,
}

/// Plugin initialization — returns a flat symbol table for JIT linking.
#[no_mangle]
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
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_plugin_describe(
    out_count: *mut usize,
) -> *const NativeMethodDesc {
    if !out_count.is_null() {
        unsafe {
            *out_count = GPU_METHODS.len();
        }
    }
    GPU_METHODS.as_ptr()
}

/// Rust-callable API returning runtime symbols.
pub fn get_runtime_symbols() -> Vec<(&'static str, *const u8)> {
    let mut symbols = vec![
        // Device lifecycle
        (
            "rayzor_gpu_compute_create",
            device::rayzor_gpu_compute_create as *const u8,
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
            ("rayzor_gpu_gfx_device_create", graphics::rayzor_gpu_gfx_device_create as *const u8),
            ("rayzor_gpu_gfx_device_destroy", graphics::rayzor_gpu_gfx_device_destroy as *const u8),
            ("rayzor_gpu_gfx_is_available", graphics::rayzor_gpu_gfx_is_available as *const u8),
            ("rayzor_gpu_gfx_buffer_create", graphics::rayzor_gpu_gfx_buffer_create as *const u8),
            ("rayzor_gpu_gfx_buffer_create_with_data", graphics::rayzor_gpu_gfx_buffer_create_with_data as *const u8),
            ("rayzor_gpu_gfx_buffer_write", graphics::rayzor_gpu_gfx_buffer_write as *const u8),
            ("rayzor_gpu_gfx_buffer_destroy", graphics::rayzor_gpu_gfx_buffer_destroy as *const u8),
            ("rayzor_gpu_gfx_shader_create", graphics::shader::rayzor_gpu_gfx_shader_create as *const u8),
            ("rayzor_gpu_gfx_shader_create_hx", graphics::shader::rayzor_gpu_gfx_shader_create_hx as *const u8),
            ("rayzor_gpu_gfx_shader_destroy", graphics::shader::rayzor_gpu_gfx_shader_destroy as *const u8),
            ("rayzor_gpu_gfx_pipeline_begin", graphics::pipeline::rayzor_gpu_gfx_pipeline_begin as *const u8),
            ("rayzor_gpu_gfx_pipeline_set_shader", graphics::pipeline::rayzor_gpu_gfx_pipeline_set_shader as *const u8),
            ("rayzor_gpu_gfx_pipeline_set_format", graphics::pipeline::rayzor_gpu_gfx_pipeline_set_format as *const u8),
            ("rayzor_gpu_gfx_pipeline_set_topology", graphics::pipeline::rayzor_gpu_gfx_pipeline_set_topology as *const u8),
            ("rayzor_gpu_gfx_pipeline_set_cull", graphics::pipeline::rayzor_gpu_gfx_pipeline_set_cull as *const u8),
            ("rayzor_gpu_gfx_pipeline_build", graphics::pipeline::rayzor_gpu_gfx_pipeline_build as *const u8),
            ("rayzor_gpu_gfx_pipeline_destroy", graphics::pipeline::rayzor_gpu_gfx_pipeline_destroy as *const u8),
            ("rayzor_gpu_gfx_texture_create", graphics::texture::rayzor_gpu_gfx_texture_create as *const u8),
            ("rayzor_gpu_gfx_texture_get_view", graphics::texture::rayzor_gpu_gfx_texture_get_view as *const u8),
            ("rayzor_gpu_gfx_texture_read_pixels", graphics::texture::rayzor_gpu_gfx_texture_read_pixels as *const u8),
            ("rayzor_gpu_gfx_texture_destroy", graphics::texture::rayzor_gpu_gfx_texture_destroy as *const u8),
            ("rayzor_gpu_gfx_render_submit", graphics::render_pass::rayzor_gpu_gfx_render_submit as *const u8),
            ("rayzor_gpu_gfx_render_triangles", graphics::render_pass::rayzor_gpu_gfx_render_triangles as *const u8),
            ("rayzor_gpu_gfx_texture_to_bytes", graphics::render_pass::rayzor_gpu_gfx_texture_to_bytes as *const u8),
        ];
        symbols.extend(gfx_symbols);
    }

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
