//! Render pipeline creation via builder pattern.

use super::bind_group::GraphicsBindGroupLayout;
use super::shader::GraphicsShader;
use super::types::*;
use super::GraphicsContext;

pub struct GraphicsPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

/// Builder for constructing a RenderPipeline incrementally.
/// Supports multiple color targets for MRT (Multiple Render Targets).
pub struct PipelineBuilder {
    shader: Option<*const GraphicsShader>,
    topology: wgpu::PrimitiveTopology,
    cull_mode: Option<wgpu::Face>,
    /// Color targets — supports MRT. First target set via setFormat(), additional via addColorTarget().
    color_targets: Vec<wgpu::TextureFormat>,
    depth_format: Option<wgpu::TextureFormat>,
    depth_compare: wgpu::CompareFunction,
    vertex_stride: u64,
    vertex_attributes: Vec<wgpu::VertexAttribute>,
    bind_group_layouts: Vec<*const GraphicsBindGroupLayout>,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[no_mangle]
pub extern "C" fn rayzor_gpu_gfx_pipeline_begin() -> *mut PipelineBuilder {
    Box::into_raw(Box::new(PipelineBuilder {
        shader: None,
        topology: wgpu::PrimitiveTopology::TriangleList,
        cull_mode: None,
        color_targets: vec![wgpu::TextureFormat::Bgra8Unorm],
        depth_format: None,
        depth_compare: wgpu::CompareFunction::Less,
        vertex_stride: 0,
        vertex_attributes: Vec::new(),
        bind_group_layouts: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_shader(
    builder: *mut PipelineBuilder,
    shader: *const GraphicsShader,
) {
    if builder.is_null() {
        return;
    }
    (*builder).shader = Some(shader);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_vertex_layout(
    builder: *mut PipelineBuilder,
    stride: u64,
    attr_count: i32,
    formats_ptr: *const i32,
    offsets_ptr: *const u64,
    locations_ptr: *const i32,
) {
    if builder.is_null() {
        return;
    }
    let b = &mut *builder;
    b.vertex_stride = stride;
    b.vertex_attributes.clear();

    for i in 0..attr_count as usize {
        let format = vertex_format_from_int(*formats_ptr.add(i));
        let offset = *offsets_ptr.add(i);
        let location = *locations_ptr.add(i) as u32;
        b.vertex_attributes.push(wgpu::VertexAttribute {
            format,
            offset,
            shader_location: location,
        });
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_format(
    builder: *mut PipelineBuilder,
    format: i32,
) {
    if builder.is_null() {
        return;
    }
    let b = &mut *builder;
    let fmt = texture_format_from_int(format);
    if b.color_targets.is_empty() {
        b.color_targets.push(fmt);
    } else {
        b.color_targets[0] = fmt;
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_topology(
    builder: *mut PipelineBuilder,
    topology: i32,
) {
    if builder.is_null() {
        return;
    }
    (*builder).topology = primitive_topology_from_int(topology);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_cull(
    builder: *mut PipelineBuilder,
    mode: i32,
) {
    if builder.is_null() {
        return;
    }
    (*builder).cull_mode = cull_mode_from_int(mode);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_set_depth(
    builder: *mut PipelineBuilder,
    format: i32,
    compare: i32,
) {
    if builder.is_null() {
        return;
    }
    (*builder).depth_format = Some(texture_format_from_int(format));
    (*builder).depth_compare = compare_function_from_int(compare);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_add_bind_group_layout(
    builder: *mut PipelineBuilder,
    layout: *const GraphicsBindGroupLayout,
) {
    if builder.is_null() || layout.is_null() {
        return;
    }
    (*builder).bind_group_layouts.push(layout);
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_build(
    builder: *mut PipelineBuilder,
    ctx: *mut GraphicsContext,
) -> *mut GraphicsPipeline {
    if builder.is_null() || ctx.is_null() {
        return std::ptr::null_mut();
    }
    let b = Box::from_raw(builder);
    let ctx = &*ctx;

    let shader = match b.shader {
        Some(s) if !s.is_null() => &*s,
        _ => return std::ptr::null_mut(),
    };

    // Build bind group layouts
    let bg_layouts: Vec<&wgpu::BindGroupLayout> = b
        .bind_group_layouts
        .iter()
        .filter_map(|&l| {
            if l.is_null() {
                None
            } else {
                Some(&(*l).layout)
            }
        })
        .collect();

    let pipeline_layout = if bg_layouts.is_empty() {
        None
    } else {
        Some(
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("rayzor_pipeline_layout"),
                    bind_group_layouts: &bg_layouts,
                    push_constant_ranges: &[],
                }),
        )
    };

    let vertex_buffers = if b.vertex_attributes.is_empty() {
        vec![]
    } else {
        vec![wgpu::VertexBufferLayout {
            array_stride: b.vertex_stride,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &b.vertex_attributes,
        }]
    };

    let depth_stencil = b.depth_format.map(|format| wgpu::DepthStencilState {
        format,
        depth_write_enabled: true,
        depth_compare: b.depth_compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });

    let color_targets: Vec<Option<wgpu::ColorTargetState>> = b
        .color_targets
        .iter()
        .map(|&fmt| {
            Some(wgpu::ColorTargetState {
                format: fmt,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })
        })
        .collect();

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rayzor_render_pipeline"),
            layout: pipeline_layout.as_ref(),
            vertex: wgpu::VertexState {
                module: &shader.module,
                entry_point: Some(&shader.vertex_entry),
                buffers: &vertex_buffers,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader.module,
                entry_point: Some(&shader.fragment_entry),
                targets: &color_targets,
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: b.topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: b.cull_mode,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    Box::into_raw(Box::new(GraphicsPipeline { pipeline }))
}

/// Add an additional color target for MRT (Multiple Render Targets).
/// The first target is set via setFormat(); this adds targets at @location(1), @location(2), etc.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_add_color_target(
    builder: *mut PipelineBuilder,
    format: i32,
) {
    if builder.is_null() {
        return;
    }
    (*builder)
        .color_targets
        .push(texture_format_from_int(format));
}

/// Get the number of color targets configured on this pipeline builder.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_get_color_target_count(
    builder: *mut PipelineBuilder,
) -> i32 {
    if builder.is_null() {
        return 0;
    }
    (*builder).color_targets.len() as i32
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_pipeline_destroy(pipeline: *mut GraphicsPipeline) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}
