//! Haxe-friendly wrappers for GPU graphics API.
//!
//! Each function accepts Haxe types (HaxeString*, HaxeBytes*, opaque pointers)
//! and translates to the internal wgpu-backed implementation.
//! These are the functions registered in the plugin symbol table and
//! called directly from Haxe extern classes.

use super::bind_group::{GraphicsBindGroup, GraphicsBindGroupLayout};
use super::pipeline::{GraphicsPipeline, PipelineBuilder};
use super::texture::{GraphicsSampler, GraphicsTexture};
use super::GraphicsBuffer;
use super::GraphicsContext;
#[cfg(feature = "native")]
use rayzor_runtime::haxe_string::HaxeString;
#[cfg(feature = "native")]
use rayzor_runtime::haxe_sys::HaxeBytes;
#[cfg(feature = "native")]
use std::ffi::c_void;

unsafe fn _hs(s: *const HaxeString) -> &'static str {
    if s.is_null() || (*s).ptr.is_null() || (*s).len == 0 {
        return "";
    }
    std::str::from_utf8(std::slice::from_raw_parts((*s).ptr, (*s).len)).unwrap_or("")
}

// ============================================================================
// Render with vertex buffer (instance method on Renderer)
// render(device, view, pipeline, vertexBuffer, vertexCount, clearR,G,B,A)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_with_vb(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_buffer: *const GraphicsBuffer,
    vertex_count: i32,
    instance_count: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    super::render_pass::rayzor_gpu_gfx_render_submit(
        ctx,
        color_view,
        0,
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        std::ptr::null(),
        pipeline,
        vertex_buffer,
        vertex_count as u32,
        instance_count as u32,
        std::ptr::null(),
        0,
        0,
        0,
        std::ptr::null(),
    );
}

// ============================================================================
// Render with vertex + index buffer
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_indexed(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_buffer: *const GraphicsBuffer,
    index_buffer: *const GraphicsBuffer,
    index_count: i32,
    instance_count: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    super::render_pass::rayzor_gpu_gfx_render_submit(
        ctx,
        color_view,
        0,
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        std::ptr::null(),
        pipeline,
        vertex_buffer,
        0, // vertex_count not used for indexed
        instance_count as u32,
        index_buffer,
        index_count as u32,
        1, // Uint32
        0,
        std::ptr::null(),
    );
}

// ============================================================================
// Render with depth buffer
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_with_depth(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    depth_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_buffer: *const GraphicsBuffer,
    vertex_count: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    super::render_pass::rayzor_gpu_gfx_render_submit(
        ctx,
        color_view,
        0,
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        depth_view,
        pipeline,
        vertex_buffer,
        vertex_count as u32,
        1,
        std::ptr::null(),
        0,
        0,
        0,
        std::ptr::null(),
    );
}

// ============================================================================
// Render with bind group (uniform buffer)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_with_bindings(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_buffer: *const GraphicsBuffer,
    vertex_count: i32,
    bind_group: *const GraphicsBindGroup,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    let bg_count = if bind_group.is_null() { 0 } else { 1 };
    let bg_ptr = if bind_group.is_null() {
        std::ptr::null()
    } else {
        &bind_group as *const *const GraphicsBindGroup
    };
    super::render_pass::rayzor_gpu_gfx_render_submit(
        ctx,
        color_view,
        0,
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        std::ptr::null(),
        pipeline,
        vertex_buffer,
        vertex_count as u32,
        1,
        std::ptr::null(),
        0,
        0,
        bg_count,
        bg_ptr,
    );
}

// ============================================================================
// Buffer create from HaxeBytes
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_from_bytes(
    ctx: *mut GraphicsContext,
    bytes: *const HaxeBytes,
    usage_flags: i32,
) -> *mut GraphicsBuffer {
    if ctx.is_null() || bytes.is_null() {
        return std::ptr::null_mut();
    }
    let b = &*bytes;
    if b.ptr.is_null() || b.len == 0 {
        return std::ptr::null_mut();
    }
    super::rayzor_gpu_gfx_buffer_create_with_data(ctx, b.ptr, b.len, usage_flags)
}

// ============================================================================
// Uniform buffer write from HaxeBytes
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_write_bytes(
    ctx: *mut GraphicsContext,
    buf: *mut GraphicsBuffer,
    offset: i64,
    bytes: *const HaxeBytes,
) {
    if ctx.is_null() || buf.is_null() || bytes.is_null() {
        return;
    }
    let b = &*bytes;
    if b.ptr.is_null() || b.len == 0 {
        return;
    }
    super::rayzor_gpu_gfx_buffer_write(ctx, buf, offset as u64, b.ptr, b.len);
}

// ============================================================================
// Bind group layout from simple spec (uniform buffer at binding 0)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_layout_uniform(
    ctx: *mut GraphicsContext,
    binding_count: i32,
) -> *mut GraphicsBindGroupLayout {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let bindings: Vec<u32> = (0..binding_count as u32).collect();
    let types: Vec<i32> = vec![0; binding_count as usize]; // 0 = UniformBuffer
    super::bind_group::rayzor_gpu_gfx_bind_group_layout_create(
        ctx,
        binding_count,
        bindings.as_ptr(),
        types.as_ptr(),
    )
}

// ============================================================================
// Bind group from single buffer (most common case)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_single(
    ctx: *mut GraphicsContext,
    layout: *const GraphicsBindGroupLayout,
    buffer: *const GraphicsBuffer,
    buffer_size: i64,
) -> *mut GraphicsBindGroup {
    if ctx.is_null() || layout.is_null() || buffer.is_null() {
        return std::ptr::null_mut();
    }
    let binding = 0u32;
    let rtype = 0i32; // Buffer
    let rptr = &(*buffer).buffer as *const wgpu::Buffer as *const c_void;
    let size = buffer_size as u64;
    super::bind_group::rayzor_gpu_gfx_bind_group_create(
        ctx, layout, 1, &binding, &rtype, &rptr, &size,
    )
}

// ============================================================================
// Pipeline with vertex layout from HaxeBytes (stride + attribute data)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_vertex_layout_simple(
    builder: *mut PipelineBuilder,
    stride: i32,
    attr_count: i32,
    // Packed: [format0, offset0, loc0, format1, offset1, loc1, ...]
    attr_data: *const i32,
) {
    if builder.is_null() || attr_data.is_null() || attr_count <= 0 {
        return;
    }
    let mut formats = Vec::new();
    let mut offsets = Vec::new();
    let mut locations = Vec::new();
    for i in 0..attr_count as usize {
        formats.push(*attr_data.add(i * 3));
        offsets.push(*attr_data.add(i * 3 + 1) as u64);
        locations.push(*attr_data.add(i * 3 + 2));
    }
    super::pipeline::rayzor_gpu_gfx_pipeline_set_vertex_layout(
        builder,
        stride as u64,
        attr_count,
        formats.as_ptr(),
        offsets.as_ptr(),
        locations.as_ptr(),
    );
}

// ============================================================================
// Pipeline with depth format
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_depth_simple(
    builder: *mut PipelineBuilder,
    depth_format: i32,
) {
    if builder.is_null() {
        return;
    }
    super::pipeline::rayzor_gpu_gfx_pipeline_set_depth(builder, depth_format, 1);
    // 1 = Less
}

// ============================================================================
// Pipeline add bind group layout
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_add_layout(
    builder: *mut PipelineBuilder,
    layout: *const GraphicsBindGroupLayout,
) {
    super::pipeline::rayzor_gpu_gfx_pipeline_add_bind_group_layout(builder, layout);
}

// ============================================================================
// Texture create for depth (convenience)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_depth_texture_create(
    ctx: *mut GraphicsContext,
    width: u32,
    height: u32,
) -> *mut GraphicsTexture {
    // Depth32Float = format 3, RENDER_ATTACHMENT = usage 16
    super::texture::rayzor_gpu_gfx_texture_create(ctx, width, height, 3, 16)
}

// ============================================================================
// Sampler create with defaults (linear filtering)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_sampler_linear(
    ctx: *mut GraphicsContext,
) -> *mut GraphicsSampler {
    // Linear mag/min, ClampToEdge
    super::texture::rayzor_gpu_gfx_sampler_create(ctx, 1, 1, 0)
}

// ============================================================================
// Texture write from HaxeBytes (for uploading image data)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_upload(
    ctx: *mut GraphicsContext,
    tex: *mut GraphicsTexture,
    bytes: *const HaxeBytes,
    bytes_per_row: i32,
) {
    if ctx.is_null() || tex.is_null() || bytes.is_null() {
        return;
    }
    let b = &*bytes;
    if b.ptr.is_null() || b.len == 0 {
        return;
    }
    super::texture::rayzor_gpu_gfx_texture_write(ctx, tex, b.ptr, b.len, bytes_per_row as u32);
}
