//! Render pipeline creation via builder pattern.

use super::bind_group::GraphicsBindGroupLayout;
#[cfg(feature = "native")]
use super::shader::GraphicsShader;
use super::types::*;
use super::GraphicsContext;

pub struct GraphicsPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

/// Builder for constructing a RenderPipeline incrementally.
/// Supports multiple color targets for MRT (Multiple Render Targets).
pub struct PipelineBuilder {
    #[cfg(feature = "native")]
    pub(crate) shader: Option<*const GraphicsShader>,
    #[cfg(not(feature = "native"))]
    pub(crate) shader: Option<i32>, // handle ID for WASM
    pub(crate) topology: wgpu::PrimitiveTopology,
    pub(crate) cull_mode: Option<wgpu::Face>,
    pub(crate) color_targets: Vec<wgpu::TextureFormat>,
    pub(crate) depth_format: Option<wgpu::TextureFormat>,
    pub(crate) depth_compare: wgpu::CompareFunction,
    pub(crate) vertex_stride: u64,
    pub(crate) vertex_attributes: Vec<wgpu::VertexAttribute>,
    pub(crate) bind_group_layouts: Vec<*const GraphicsBindGroupLayout>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            shader: None,
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            color_targets: vec![],
            depth_format: None,
            depth_compare: wgpu::CompareFunction::Less,
            vertex_stride: 0,
            vertex_attributes: vec![],
            bind_group_layouts: vec![],
        }
    }

    pub fn set_vertex_layout_simple(&mut self, stride: i32, attr_count: i32, attr_data: &[i32]) {
        self.vertex_stride = stride as u64;
        self.vertex_attributes.clear();
        let mut offset = 0u64;
        for i in 0..attr_count as usize {
            if i >= attr_data.len() {
                break;
            }
            let components = attr_data[i];
            let format = match components {
                1 => wgpu::VertexFormat::Float32,
                2 => wgpu::VertexFormat::Float32x2,
                3 => wgpu::VertexFormat::Float32x3,
                4 => wgpu::VertexFormat::Float32x4,
                _ => wgpu::VertexFormat::Float32x4,
            };
            self.vertex_attributes.push(wgpu::VertexAttribute {
                format,
                offset,
                shader_location: i as u32,
            });
            offset += (components as u64) * 4;
        }
    }

    pub fn set_depth_simple(&mut self, depth_format: i32) {
        self.depth_format = Some(super::types::int_to_texture_format(depth_format));
    }

    pub fn add_layout(&mut self, layout: *const super::bind_group::GraphicsBindGroupLayout) {
        self.bind_group_layouts.push(layout);
    }
}

// ============================================================================
// Extern "C" entry points (native only)
// ============================================================================
#[cfg(feature = "native")]
mod native_ffi {
    use super::*;

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

    #[cfg(feature = "native")]
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
} // mod native_ffi
#[cfg(feature = "native")]
pub use native_ffi::*;
