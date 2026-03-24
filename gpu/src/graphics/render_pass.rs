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

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_encoder_create(
    ctx: *mut GraphicsContext,
) -> *mut GraphicsEncoder {
    if ctx.is_null() { return std::ptr::null_mut(); }
    let ctx = &*ctx;
    let encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rayzor_encoder"),
    });
    Box::into_raw(Box::new(GraphicsEncoder { encoder: Some(encoder) }))
}

/// Begin a render pass. The pass borrows the encoder, so we use a simplified
/// approach: store the color attachment info and create the pass inline.
///
/// For the initial implementation, we provide a single-pass submit function
/// that handles the full lifecycle: begin pass → record commands → end → submit.
///
/// This avoids the complex lifetime issues of exposing wgpu::RenderPass across FFI.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_render_submit(
    ctx: *mut GraphicsContext,
    color_view: *const wgpu::TextureView,
    load_op: i32,
    clear_r: f64, clear_g: f64, clear_b: f64, clear_a: f64,
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

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_encoder_destroy(encoder: *mut GraphicsEncoder) {
    if !encoder.is_null() {
        drop(Box::from_raw(encoder));
    }
}
