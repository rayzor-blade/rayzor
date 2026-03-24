//! Bind groups for resource binding (uniforms, textures, samplers).

use super::texture::{GraphicsSampler, GraphicsTexture};
use super::GraphicsContext;

pub struct GraphicsBindGroupLayout {
    pub layout: wgpu::BindGroupLayout,
}

pub struct GraphicsBindGroup {
    pub bind_group: wgpu::BindGroup,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

/// Create a bind group layout.
/// `entry_types` array: 0=UniformBuffer, 1=StorageBuffer, 2=Texture, 3=Sampler
/// `entry_bindings` array: binding index for each entry
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_layout_create(
    ctx: *mut GraphicsContext,
    entry_count: i32,
    entry_bindings: *const u32,
    entry_types: *const i32,
) -> *mut GraphicsBindGroupLayout {
    if ctx.is_null() { return std::ptr::null_mut(); }
    let ctx = &*ctx;

    let visibility = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let mut entries = Vec::new();

    for i in 0..entry_count as usize {
        let binding = *entry_bindings.add(i);
        let ty = *entry_types.add(i);

        let binding_type = match ty {
            0 => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            1 => wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            2 => wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            3 => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            _ => continue,
        };

        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: binding_type,
            count: None,
        });
    }

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rayzor_bind_group_layout"),
        entries: &entries,
    });

    Box::into_raw(Box::new(GraphicsBindGroupLayout { layout }))
}

/// Create a bind group from a layout and resource bindings.
/// `resource_types`: 0=Buffer, 1=TextureView, 2=Sampler
/// `resource_ptrs`: pointer to the resource
/// `buffer_offsets` / `buffer_sizes`: only used for buffer resources
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_create(
    ctx: *mut GraphicsContext,
    layout: *const GraphicsBindGroupLayout,
    entry_count: i32,
    entry_bindings: *const u32,
    resource_types: *const i32,
    resource_ptrs: *const *const std::ffi::c_void,
    buffer_sizes: *const u64,
) -> *mut GraphicsBindGroup {
    if ctx.is_null() || layout.is_null() { return std::ptr::null_mut(); }
    let ctx = &*ctx;
    let layout = &*layout;

    let mut entries = Vec::new();

    for i in 0..entry_count as usize {
        let binding = *entry_bindings.add(i);
        let rtype = *resource_types.add(i);
        let ptr = *resource_ptrs.add(i);

        let resource = match rtype {
            0 => {
                // Buffer
                let buffer = &*(ptr as *const wgpu::Buffer);
                let size = *buffer_sizes.add(i);
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: if size > 0 {
                        std::num::NonZeroU64::new(size).map(wgpu::BufferSize::from)
                    } else {
                        None
                    },
                })
            }
            1 => {
                // TextureView
                let view = &*(ptr as *const wgpu::TextureView);
                wgpu::BindingResource::TextureView(view)
            }
            2 => {
                // Sampler
                let sampler = &*(ptr as *const GraphicsSampler);
                wgpu::BindingResource::Sampler(&sampler.sampler)
            }
            _ => continue,
        };

        entries.push(wgpu::BindGroupEntry { binding, resource });
    }

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rayzor_bind_group"),
        layout: &layout.layout,
        entries: &entries,
    });

    Box::into_raw(Box::new(GraphicsBindGroup { bind_group }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_destroy(bg: *mut GraphicsBindGroup) {
    if !bg.is_null() {
        drop(Box::from_raw(bg));
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_bind_group_layout_destroy(
    layout: *mut GraphicsBindGroupLayout,
) {
    if !layout.is_null() {
        drop(Box::from_raw(layout));
    }
}
