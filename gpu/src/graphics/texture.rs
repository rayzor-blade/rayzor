//! Texture and sampler creation.

use super::types::*;
use super::GraphicsContext;

pub struct GraphicsTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

pub struct GraphicsSampler {
    pub sampler: wgpu::Sampler,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_create(
    ctx: *mut GraphicsContext,
    width: u32,
    height: u32,
    format: i32,
    usage_flags: i32,
) -> *mut GraphicsTexture {
    if ctx.is_null() { return std::ptr::null_mut(); }
    let ctx = &*ctx;

    let tex_format = texture_format_from_int(format);
    let usage = texture_usages_from_flags(usage_flags);

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rayzor_texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: tex_format,
        usage,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    Box::into_raw(Box::new(GraphicsTexture { texture, view, width, height }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_write(
    ctx: *mut GraphicsContext,
    tex: *mut GraphicsTexture,
    data_ptr: *const u8,
    data_len: usize,
    bytes_per_row: u32,
) {
    if ctx.is_null() || tex.is_null() || data_ptr.is_null() { return; }
    let ctx = &*ctx;
    let tex = &*tex;
    let data = std::slice::from_raw_parts(data_ptr, data_len);

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(tex.height),
        },
        wgpu::Extent3d { width: tex.width, height: tex.height, depth_or_array_layers: 1 },
    );
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_get_view(
    tex: *const GraphicsTexture,
) -> *const wgpu::TextureView {
    if tex.is_null() { return std::ptr::null(); }
    &(*tex).view as *const wgpu::TextureView
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_destroy(tex: *mut GraphicsTexture) {
    if !tex.is_null() {
        drop(Box::from_raw(tex));
    }
}

/// Read pixel data from a texture into a CPU buffer.
/// The texture must have COPY_SRC usage. Returns the number of bytes written,
/// or 0 on failure. Output is RGBA8 (4 bytes per pixel).
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_read_pixels(
    ctx: *mut GraphicsContext,
    tex: *mut GraphicsTexture,
    out_ptr: *mut u8,
    out_capacity: usize,
) -> usize {
    if ctx.is_null() || tex.is_null() || out_ptr.is_null() {
        return 0;
    }
    let ctx = &*ctx;
    let tex = &*tex;

    let bytes_per_pixel = 4u32; // RGBA8
    let padded_bytes_per_row = (tex.width * bytes_per_pixel + 255) & !255; // Align to 256
    let total_size = (padded_bytes_per_row * tex.height) as u64;

    // Create staging buffer
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rayzor_readback_staging"),
        size: total_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture → staging buffer
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rayzor_readback_encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(tex.height),
            },
        },
        wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Map and read back
    let buffer_slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    ctx.device.poll(wgpu::Maintain::Wait);

    if receiver.recv().ok().and_then(|r| r.ok()).is_none() {
        return 0;
    }

    let mapped = buffer_slice.get_mapped_range();
    let unpadded_row_bytes = (tex.width * bytes_per_pixel) as usize;
    let needed = unpadded_row_bytes * tex.height as usize;
    if out_capacity < needed {
        return 0;
    }

    // Copy with padding removal
    let out = std::slice::from_raw_parts_mut(out_ptr, needed);
    for row in 0..tex.height as usize {
        let src_offset = row * padded_bytes_per_row as usize;
        let dst_offset = row * unpadded_row_bytes;
        out[dst_offset..dst_offset + unpadded_row_bytes]
            .copy_from_slice(&mapped[src_offset..src_offset + unpadded_row_bytes]);
    }

    drop(mapped);
    staging.unmap();
    needed
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_sampler_create(
    ctx: *mut GraphicsContext,
    mag_filter: i32,
    min_filter: i32,
    address_mode: i32,
) -> *mut GraphicsSampler {
    if ctx.is_null() { return std::ptr::null_mut(); }
    let ctx = &*ctx;

    let addr = address_mode_from_int(address_mode);
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("rayzor_sampler"),
        mag_filter: filter_mode_from_int(mag_filter),
        min_filter: filter_mode_from_int(min_filter),
        address_mode_u: addr,
        address_mode_v: addr,
        address_mode_w: addr,
        ..Default::default()
    });

    Box::into_raw(Box::new(GraphicsSampler { sampler }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_sampler_destroy(sampler: *mut GraphicsSampler) {
    if !sampler.is_null() {
        drop(Box::from_raw(sampler));
    }
}
