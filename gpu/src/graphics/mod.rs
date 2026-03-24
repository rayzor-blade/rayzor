//! GPU Graphics Rendering — wgpu-backed render pipeline.
//!
//! Provides graphics rendering capabilities mirroring the WebGPU API:
//! surfaces, shader modules, render pipelines, textures, bind groups,
//! and command encoding. Used from Haxe via extern native methods.

pub mod bind_group;
pub mod command;
pub mod haxe_api;
pub mod pipeline;
pub mod render_pass;
pub mod shader;
pub mod surface;
pub mod texture;
pub mod types;

use std::sync::Arc;

/// Graphics context wrapping wgpu Device + Queue + Instance.
/// Persists for the lifetime of the application.
pub struct GraphicsContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
}

impl GraphicsContext {
    /// Create a new graphics context (headless — no surface required).
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rayzor_graphics"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        Some(GraphicsContext {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    pub fn is_available() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_some()
    }
}

// ============================================================================
// Extern "C" entry points for Haxe FFI
// ============================================================================

#[no_mangle]
pub extern "C" fn rayzor_gpu_gfx_device_create() -> *mut GraphicsContext {
    match GraphicsContext::new() {
        Some(ctx) => Box::into_raw(Box::new(ctx)),
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_device_destroy(ctx: *mut GraphicsContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

#[no_mangle]
pub extern "C" fn rayzor_gpu_gfx_is_available() -> i32 {
    if GraphicsContext::is_available() {
        1
    } else {
        0
    }
}

// ============================================================================
// Graphics buffer operations
// ============================================================================

pub struct GraphicsBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_create(
    ctx: *mut GraphicsContext,
    size: u64,
    usage_flags: i32,
) -> *mut GraphicsBuffer {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;
    let usage = types::buffer_usages_from_flags(usage_flags) | wgpu::BufferUsages::COPY_DST;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rayzor_buffer"),
        size,
        usage,
        mapped_at_creation: false,
    });

    Box::into_raw(Box::new(GraphicsBuffer { buffer, size }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_write(
    ctx: *mut GraphicsContext,
    buf: *mut GraphicsBuffer,
    offset: u64,
    data_ptr: *const u8,
    data_len: usize,
) {
    if ctx.is_null() || buf.is_null() || data_ptr.is_null() {
        return;
    }
    let ctx = &*ctx;
    let buf = &*buf;
    let data = std::slice::from_raw_parts(data_ptr, data_len);
    ctx.queue.write_buffer(&buf.buffer, offset, data);
}

/// Create a buffer and upload data in one call. Convenience for vertex/index data.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_create_with_data(
    ctx: *mut GraphicsContext,
    data_ptr: *const u8,
    data_len: usize,
    usage_flags: i32,
) -> *mut GraphicsBuffer {
    if ctx.is_null() || data_ptr.is_null() || data_len == 0 {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;
    let data = std::slice::from_raw_parts(data_ptr, data_len);
    let usage = types::buffer_usages_from_flags(usage_flags) | wgpu::BufferUsages::COPY_DST;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rayzor_buffer_init"),
        size: data_len as u64,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(data);
    buffer.unmap();

    Box::into_raw(Box::new(GraphicsBuffer {
        buffer,
        size: data_len as u64,
    }))
}

/// Get the inner wgpu::Buffer pointer for interop (e.g., compute → graphics).
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_get_inner(
    buf: *const GraphicsBuffer,
) -> *const wgpu::Buffer {
    if buf.is_null() {
        return std::ptr::null();
    }
    &(*buf).buffer as *const wgpu::Buffer
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_buffer_destroy(buf: *mut GraphicsBuffer) {
    if !buf.is_null() {
        drop(Box::from_raw(buf));
    }
}
