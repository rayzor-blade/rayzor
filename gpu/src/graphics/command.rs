//! Multi-pass command encoder with deferred render pass recording.
//!
//! Records render commands as a Vec<RenderCommand>, then replays them
//! into a real wgpu::RenderPass at submit time. This avoids the lifetime
//! issues of exposing wgpu::RenderPass across the FFI boundary.

use super::bind_group::GraphicsBindGroup;
use super::pipeline::GraphicsPipeline;
use super::texture::GraphicsTexture;
use super::GraphicsBuffer;
use super::GraphicsContext;

/// A recorded render command.
enum RenderCommand {
    SetPipeline(*const GraphicsPipeline),
    SetVertexBuffer(u32, *const GraphicsBuffer),
    SetIndexBuffer(*const GraphicsBuffer, wgpu::IndexFormat),
    SetBindGroup(u32, *const GraphicsBindGroup),
    Draw(u32, u32, u32, u32), // vertex_count, instance_count, first_vertex, first_instance
    DrawIndexed(u32, u32, u32, i32, u32), // index_count, instance_count, first_index, base_vertex, first_instance
    SetViewport(f32, f32, f32, f32, f32, f32), // x, y, w, h, min_depth, max_depth
    SetScissor(u32, u32, u32, u32),       // x, y, w, h
}

/// A recorded render pass — stores attachment config + commands.
pub struct RecordedRenderPass {
    color_view: *const wgpu::TextureView,
    load_op: wgpu::LoadOp<wgpu::Color>,
    depth_view: Option<*const wgpu::TextureView>,
    commands: Vec<RenderCommand>,
}

/// Command encoder that records multiple render passes.
pub struct CommandRecorder {
    passes: Vec<RecordedRenderPass>,
    current_pass: Option<RecordedRenderPass>,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[no_mangle]
pub extern "C" fn rayzor_gpu_gfx_cmd_create() -> *mut CommandRecorder {
    Box::into_raw(Box::new(CommandRecorder {
        passes: Vec::new(),
        current_pass: None,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_begin_pass(
    cmd: *mut CommandRecorder,
    color_view: *const wgpu::TextureView,
    load_op: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
    depth_view: *const wgpu::TextureView,
) {
    if cmd.is_null() || color_view.is_null() {
        return;
    }
    let cmd = &mut *cmd;

    // End current pass if any
    if let Some(pass) = cmd.current_pass.take() {
        cmd.passes.push(pass);
    }

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

    cmd.current_pass = Some(RecordedRenderPass {
        color_view,
        load_op: load,
        depth_view: if depth_view.is_null() {
            None
        } else {
            Some(depth_view)
        },
        commands: Vec::new(),
    });
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_pipeline(
    cmd: *mut CommandRecorder,
    pipeline: *const GraphicsPipeline,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands.push(RenderCommand::SetPipeline(pipeline));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_vertex_buffer(
    cmd: *mut CommandRecorder,
    slot: u32,
    buffer: *const GraphicsBuffer,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands
            .push(RenderCommand::SetVertexBuffer(slot, buffer));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_index_buffer(
    cmd: *mut CommandRecorder,
    buffer: *const GraphicsBuffer,
    format: i32,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        let fmt = if format == 1 {
            wgpu::IndexFormat::Uint32
        } else {
            wgpu::IndexFormat::Uint16
        };
        pass.commands
            .push(RenderCommand::SetIndexBuffer(buffer, fmt));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_bind_group(
    cmd: *mut CommandRecorder,
    group_index: u32,
    bind_group: *const GraphicsBindGroup,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands
            .push(RenderCommand::SetBindGroup(group_index, bind_group));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_draw(
    cmd: *mut CommandRecorder,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands.push(RenderCommand::Draw(
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        ));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_draw_indexed(
    cmd: *mut CommandRecorder,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands.push(RenderCommand::DrawIndexed(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        ));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_viewport(
    cmd: *mut CommandRecorder,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    min_depth: f32,
    max_depth: f32,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands
            .push(RenderCommand::SetViewport(x, y, w, h, min_depth, max_depth));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_set_scissor(
    cmd: *mut CommandRecorder,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    if let Some(pass) = (*cmd).current_pass.as_mut() {
        pass.commands.push(RenderCommand::SetScissor(x, y, w, h));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_end_pass(cmd: *mut CommandRecorder) {
    if cmd.is_null() {
        return;
    }
    let cmd = &mut *cmd;
    if let Some(pass) = cmd.current_pass.take() {
        cmd.passes.push(pass);
    }
}

/// Submit all recorded passes. Consumes the recorder.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_submit(
    cmd: *mut CommandRecorder,
    ctx: *mut GraphicsContext,
) {
    if cmd.is_null() || ctx.is_null() {
        return;
    }
    let mut cmd = Box::from_raw(cmd);
    let ctx = &*ctx;

    // End any dangling pass
    if let Some(pass) = cmd.current_pass.take() {
        cmd.passes.push(pass);
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rayzor_multi_pass"),
        });

    for recorded_pass in &cmd.passes {
        let depth_attachment =
            recorded_pass
                .depth_view
                .map(|dv| wgpu::RenderPassDepthStencilAttachment {
                    view: &*dv,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rayzor_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &*recorded_pass.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: recorded_pass.load_op,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: depth_attachment,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Replay recorded commands
            for command in &recorded_pass.commands {
                match command {
                    RenderCommand::SetPipeline(p) => {
                        pass.set_pipeline(&(**p).pipeline);
                    }
                    RenderCommand::SetVertexBuffer(slot, buf) => {
                        pass.set_vertex_buffer(*slot, (**buf).buffer.slice(..));
                    }
                    RenderCommand::SetIndexBuffer(buf, fmt) => {
                        pass.set_index_buffer((**buf).buffer.slice(..), *fmt);
                    }
                    RenderCommand::SetBindGroup(idx, bg) => {
                        pass.set_bind_group(*idx, Some(&(**bg).bind_group), &[]);
                    }
                    RenderCommand::Draw(vc, ic, fv, fi) => {
                        pass.draw(0..*vc, 0..*ic);
                        let _ = (fv, fi); // TODO: use first_vertex/first_instance overloads
                    }
                    RenderCommand::DrawIndexed(ic, inst, fi, bv, finst) => {
                        pass.draw_indexed(0..*ic, *bv, 0..*inst);
                        let _ = (fi, finst);
                    }
                    RenderCommand::SetViewport(x, y, w, h, mind, maxd) => {
                        pass.set_viewport(*x, *y, *w, *h, *mind, *maxd);
                    }
                    RenderCommand::SetScissor(x, y, w, h) => {
                        pass.set_scissor_rect(*x, *y, *w, *h);
                    }
                }
            }
        } // pass dropped, releasing encoder borrow
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_cmd_destroy(cmd: *mut CommandRecorder) {
    if !cmd.is_null() {
        drop(Box::from_raw(cmd));
    }
}
