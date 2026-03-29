//! Command encoder and render pass recording.

use super::bind_group::GraphicsBindGroup;
use super::pipeline::GraphicsPipeline;
use super::GraphicsBuffer;
use super::GraphicsContext;

/// Wrapper around wgpu CommandEncoder. Owns the encoder until finish+submit.
pub struct GraphicsEncoder {
    pub encoder: Option<wgpu::CommandEncoder>,
}

/// Opaque render pass handle. The actual wgpu::RenderPass has a borrow on
/// the encoder, so we store it as raw parts and manage lifetime manually.
pub struct GraphicsRenderPass {
    // We store the render pass as a boxed trait object to erase the lifetime.
    // The pass MUST be dropped before the encoder is finished.
    _marker: std::marker::PhantomData<()>,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_encoder_create(
    ctx: *mut GraphicsContext,
) -> *mut GraphicsEncoder {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;
    let encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rayzor_encoder"),
        });
    Box::into_raw(Box::new(GraphicsEncoder {
        encoder: Some(encoder),
    }))
}

/// Begin a render pass. The pass borrows the encoder, so we use a simplified
/// approach: store the color attachment info and create the pass inline.
///
/// For the initial implementation, we provide a single-pass submit function
/// that handles the full lifecycle: begin pass → record commands → end → submit.
///
/// This avoids the complex lifetime issues of exposing wgpu::RenderPass across FFI.
#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_submit(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    load_op: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
    depth_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_buffer: *const GraphicsBuffer,
    vertex_count: u32,
    instance_count: u32,
    index_buffer: *const GraphicsBuffer,
    index_count: u32,
    index_format: i32,
    bind_group_count: i32,
    bind_groups: *const *const GraphicsBindGroup,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    let ctx = &*ctx;
    let color_view = &*color_view;
    let pipeline = &*pipeline;

    let load = if load_op == 0 {
        wgpu::LoadOp::Clear(wgpu::Color {
            r: clear_r,
            g: clear_g,
            b: clear_b,
            a: clear_a,
        })
    } else {
        wgpu::LoadOp::Load
    };

    let depth_attachment = if !depth_view.is_null() {
        Some(wgpu::RenderPassDepthStencilAttachment {
            view: &*depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        })
    } else {
        None
    };

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rayzor_render"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("rayzor_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&pipeline.pipeline);

        // Bind groups
        for i in 0..bind_group_count as usize {
            let bg_ptr = *bind_groups.add(i);
            if !bg_ptr.is_null() {
                pass.set_bind_group(i as u32, Some(&(*bg_ptr).bind_group), &[]);
            }
        }

        // Vertex buffer
        if !vertex_buffer.is_null() {
            pass.set_vertex_buffer(0, (*vertex_buffer).buffer.slice(..));
        }

        // Index buffer + indexed draw
        if !index_buffer.is_null() && index_count > 0 {
            let fmt = if index_format == 1 {
                wgpu::IndexFormat::Uint32
            } else {
                wgpu::IndexFormat::Uint16
            };
            pass.set_index_buffer((*index_buffer).buffer.slice(..), fmt);
            pass.draw_indexed(0..index_count, 0, 0..instance_count);
        } else if vertex_count > 0 {
            pass.draw(0..vertex_count, 0..instance_count);
        }
    } // pass dropped here, releasing borrow on encoder

    ctx.queue.submit(std::iter::once(encoder.finish()));
}

/// Haxe-friendly: render triangle(s) with clear color, no vertex buffer needed.
/// Simpler than render_submit — covers the common @:shader procedural case.
#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_triangles(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    pipeline: *const GraphicsPipeline,
    vertex_count: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
) {
    if ctx.is_null() || color_view.is_null() || pipeline.is_null() {
        return;
    }
    rayzor_gpu_gfx_render_submit(
        ctx,
        color_view,
        0, // Clear
        clear_r,
        clear_g,
        clear_b,
        clear_a,
        std::ptr::null(), // no depth
        pipeline,
        std::ptr::null(), // no vertex buffer
        vertex_count as u32,
        1,                // 1 instance
        std::ptr::null(), // no index buffer
        0,
        0,
        0,
        std::ptr::null(), // no bind groups
    );
}

/// Haxe-friendly: read texture pixels into a newly allocated buffer.
/// Returns pointer to RGBA8 data (caller must free), sets out_len.
#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_read_rgba(
    ctx: *mut GraphicsContext,
    tex: *mut super::texture::GraphicsTexture,
    out_len: *mut usize,
) -> *mut u8 {
    if ctx.is_null() || tex.is_null() {
        return std::ptr::null_mut();
    }
    let tex_ref = &*tex;
    let byte_count = (tex_ref.width * tex_ref.height * 4) as usize;
    let mut buf = vec![0u8; byte_count];
    let read =
        super::texture::rayzor_gpu_gfx_texture_read_pixels(ctx, tex, buf.as_mut_ptr(), buf.len());
    if read == 0 {
        return std::ptr::null_mut();
    }
    if !out_len.is_null() {
        *out_len = read;
    }
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf); // caller owns the memory
    ptr
}

/// Haxe-friendly: read texture pixels into a HaxeBytes (RGBA8, 4 bytes per pixel).
/// Returns null on failure.
#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_texture_to_bytes(
    tex: *mut super::texture::GraphicsTexture,
    ctx: *mut GraphicsContext,
) -> *mut rayzor_runtime::haxe_sys::HaxeBytes {
    use rayzor_runtime::haxe_sys::haxe_bytes_alloc;

    if ctx.is_null() || tex.is_null() {
        return std::ptr::null_mut();
    }
    let tex_ref = &*tex;
    let byte_count = (tex_ref.width * tex_ref.height * 4) as i32;
    let bytes = haxe_bytes_alloc(byte_count);
    if bytes.is_null() {
        return std::ptr::null_mut();
    }

    let read =
        super::texture::rayzor_gpu_gfx_texture_read_pixels(ctx, tex, (*bytes).ptr, (*bytes).len);
    if read == 0 {
        // Failed — free and return null
        return std::ptr::null_mut();
    }

    bytes
}

/// Haxe-friendly: read texture pixels and save as PPM file.
/// Accepts a HaxeString* path.
#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_save_texture_ppm(
    ctx: *mut GraphicsContext,
    tex: *mut super::texture::GraphicsTexture,
    path: *const rayzor_runtime::haxe_string::HaxeString,
) -> i32 {
    if ctx.is_null() || tex.is_null() || path.is_null() {
        return 0;
    }
    let path_str = {
        let hs = &*path;
        if hs.ptr.is_null() || hs.len == 0 {
            return 0;
        }
        std::str::from_utf8(std::slice::from_raw_parts(hs.ptr, hs.len)).unwrap_or("output.ppm")
    };

    let tex_ref = &*tex;
    let w = tex_ref.width;
    let h = tex_ref.height;
    let byte_count = (w * h * 4) as usize;
    let mut pixels = vec![0u8; byte_count];
    let read = super::texture::rayzor_gpu_gfx_texture_read_pixels(
        ctx,
        tex,
        pixels.as_mut_ptr(),
        pixels.len(),
    );
    if read == 0 {
        return 0;
    }

    // Build PPM (P6 binary)
    let header = format!("P6\n{} {}\n255\n", w, h);
    let mut data = header.into_bytes();
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            data.push(pixels[i]); // R
            data.push(pixels[i + 1]); // G
            data.push(pixels[i + 2]); // B
        }
    }

    match std::fs::write(path_str, &data) {
        Ok(_) => 1,
        Err(_) => 0,
    }
}

#[cfg(feature = "native")]
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_encoder_destroy(encoder: *mut GraphicsEncoder) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}
