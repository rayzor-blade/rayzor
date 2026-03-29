//! WASM host exports via wasm-bindgen.
//!
//! Full feature parity with native extern "C" functions.
//! Uses a handle table for cross-WASM-boundary safety.
//!
//! Build: wasm-pack build --target web --no-default-features --features wasm-host

use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use wasm_bindgen::prelude::*;

use crate::graphics::bind_group::{GraphicsBindGroup, GraphicsBindGroupLayout};
use crate::graphics::command::CommandRecorder;
use crate::graphics::pipeline::{GraphicsPipeline, PipelineBuilder};
use crate::graphics::surface::GraphicsSurface;
use crate::graphics::texture::{GraphicsSampler, GraphicsTexture};
use crate::graphics::types;
use crate::graphics::GraphicsContext;

// ============================================================================
// Handle table
// ============================================================================

static HANDLES: Lazy<Mutex<HandleTable>> = Lazy::new(|| Mutex::new(HandleTable::new()));

struct HandleTable {
    next: i32,
    objects: HashMap<i32, GpuObject>,
}

#[allow(dead_code)]
enum GpuObject {
    GfxContext(Box<GraphicsContext>),
    Surface(Box<GraphicsSurface>),
    Pipeline(Box<GraphicsPipeline>),
    PipelineBuilder(Box<PipelineBuilder>),
    Texture(Box<GraphicsTexture>),
    TextureView(Box<wgpu::TextureView>),
    Buffer(Box<crate::graphics::GraphicsBuffer>),
    Sampler(Box<GraphicsSampler>),
    BindGroup(Box<GraphicsBindGroup>),
    BindGroupLayout(Box<GraphicsBindGroupLayout>),
    CommandRecorder(Box<CommandRecorder>),
    Shader {
        module: wgpu::ShaderModule,
        vs_entry: String,
        fs_entry: String,
    },
}

// SAFETY: WASM is single-threaded. The handle table is only accessed from the main thread.
unsafe impl Send for HandleTable {}
unsafe impl Sync for HandleTable {}

impl HandleTable {
    fn new() -> Self {
        Self {
            next: 1,
            objects: HashMap::new(),
        }
    }
    fn alloc(&mut self, obj: GpuObject) -> i32 {
        let h = self.next;
        self.next += 1;
        self.objects.insert(h, obj);
        h
    }
    #[allow(dead_code)]
    fn get(&self, h: i32) -> Option<&GpuObject> {
        self.objects.get(&h)
    }
    #[allow(dead_code)]
    fn get_mut(&mut self, h: i32) -> Option<&mut GpuObject> {
        self.objects.get_mut(&h)
    }
    fn remove(&mut self, h: i32) -> Option<GpuObject> {
        self.objects.remove(&h)
    }
    fn free(&mut self, h: i32) {
        self.objects.remove(&h);
    }
}

// ============================================================================
// Device
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_device_create")]
pub async fn gfx_device_create() -> i32 {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
    {
        Some(a) => a,
        None => return 0,
    };
    let (device, queue) = match adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rayzor_gpu_wasm"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                ..Default::default()
            },
            None,
        )
        .await
    {
        Ok(dq) => dq,
        Err(_) => return 0,
    };
    let ctx = GraphicsContext {
        instance,
        adapter,
        device: std::sync::Arc::new(device),
        queue: std::sync::Arc::new(queue),
    };
    HANDLES
        .lock()
        .unwrap()
        .alloc(GpuObject::GfxContext(Box::new(ctx)))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_device_destroy")]
pub fn gfx_device_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_is_available")]
pub fn gfx_is_available() -> i32 {
    #[cfg(target_arch = "wasm32")]
    {
        if web_sys::window()
            .and_then(|w| js_sys::Reflect::get(&w.navigator(), &"gpu".into()).ok())
            .map(|v| !v.is_undefined())
            .unwrap_or(false)
        {
            return 1;
        }
    }
    0
}

// ============================================================================
// Surface (canvas-based on WASM)
// ============================================================================

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_create_canvas")]
pub fn gfx_surface_create_canvas(dev_h: i32, canvas_id: &str, width: i32, height: i32) -> i32 {
    use wasm_bindgen::JsCast;
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };

    let doc = match web_sys::window().and_then(|w| w.document()) {
        Some(d) => d,
        None => return 0,
    };

    let canvas: web_sys::HtmlCanvasElement = if !canvas_id.is_empty() {
        doc.get_element_by_id(canvas_id)
            .and_then(|e| e.dyn_into().ok())
            .unwrap_or_else(|| {
                let c: web_sys::HtmlCanvasElement =
                    doc.create_element("canvas").unwrap().dyn_into().unwrap();
                c.set_id(canvas_id);
                doc.body().unwrap().append_child(&c).unwrap();
                c
            })
    } else {
        doc.query_selector("canvas")
            .ok()
            .flatten()
            .and_then(|e| e.dyn_into().ok())
            .unwrap_or_else(|| {
                let c: web_sys::HtmlCanvasElement =
                    doc.create_element("canvas").unwrap().dyn_into().unwrap();
                c.set_id("rayzor-canvas");
                doc.body().unwrap().append_child(&c).unwrap();
                c
            })
    };
    canvas.set_width(width.max(1) as u32);
    canvas.set_height(height.max(1) as u32);

    let surface = match ctx
        .instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
    {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let caps = surface.get_capabilities(&ctx.adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1) as u32,
        height: height.max(1) as u32,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&ctx.device, &config);
    ht.alloc(GpuObject::Surface(Box::new(GraphicsSurface {
        surface,
        config,
        format,
        current_texture: None,
    })))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_get_texture")]
pub fn gfx_surface_get_texture(h: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let surf = match ht.get_mut(h) {
        Some(GpuObject::Surface(s)) => s,
        _ => return 0,
    };
    match surf.surface.get_current_texture() {
        Ok(frame) => {
            let view = frame.texture.create_view(&Default::default());
            surf.current_texture = Some(frame);
            ht.alloc(GpuObject::TextureView(Box::new(view)))
        }
        Err(_) => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_present")]
pub fn gfx_surface_present(h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::Surface(s)) = ht.get_mut(h) {
        if let Some(tex) = s.current_texture.take() {
            tex.present();
        }
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_resize")]
pub fn gfx_surface_resize(h: i32, dev_h: i32, w: i32, ht_val: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let device = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c.device.clone(),
        _ => return,
    };
    if let Some(GpuObject::Surface(s)) = ht.get_mut(h) {
        s.config.width = w.max(1) as u32;
        s.config.height = ht_val.max(1) as u32;
        s.surface.configure(&device, &s.config);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_get_format")]
pub fn gfx_surface_get_format(h: i32) -> i32 {
    let ht = HANDLES.lock().unwrap();
    match ht.get(h) {
        Some(GpuObject::Surface(s)) => types::texture_format_to_int(s.format),
        _ => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_destroy")]
pub fn gfx_surface_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Shader
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_shader_create_hx")]
pub fn gfx_shader_create(dev_h: i32, wgsl: &str, vs: &str, fs: &str) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    ht.alloc(GpuObject::Shader {
        module,
        vs_entry: if vs.is_empty() {
            "vs_main".into()
        } else {
            vs.into()
        },
        fs_entry: if fs.is_empty() {
            "fs_main".into()
        } else {
            fs.into()
        },
    })
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_shader_destroy")]
pub fn gfx_shader_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Buffer
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_create")]
pub fn gfx_buffer_create(dev_h: i32, size: i32, usage: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size.max(4) as u64,
        usage: wgpu::BufferUsages::from_bits_truncate(usage as u32),
        mapped_at_creation: false,
    });
    ht.alloc(GpuObject::Buffer(Box::new(
        crate::graphics::GraphicsBuffer {
            buffer,
            size: size as u64,
        },
    )))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_create_with_data")]
pub fn gfx_buffer_create_with_data(dev_h: i32, data: &[u8], usage: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    use wgpu::util::DeviceExt;
    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: wgpu::BufferUsages::from_bits_truncate(usage as u32),
        });
    ht.alloc(GpuObject::Buffer(Box::new(
        crate::graphics::GraphicsBuffer {
            buffer,
            size: data.len() as u64,
        },
    )))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_write")]
pub fn gfx_buffer_write(buf_h: i32, dev_h: i32, offset: i32, data: &[u8]) {
    let ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return,
    };
    if let Some(GpuObject::Buffer(b)) = ht.get(buf_h) {
        ctx.queue.write_buffer(&b.buffer, offset as u64, data);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_destroy")]
pub fn gfx_buffer_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Texture
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_texture_create")]
pub fn gfx_texture_create(dev_h: i32, w: i32, h: i32, fmt: i32, usage: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let format = types::int_to_texture_format(fmt);
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: w as u32,
            height: h as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::from_bits_truncate(usage as u32),
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    ht.alloc(GpuObject::Texture(Box::new(GraphicsTexture {
        texture,
        view,
        width: w as u32,
        height: h as u32,
    })))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_texture_get_view")]
pub fn gfx_texture_get_view(h: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let view = match ht.get(h) {
        Some(GpuObject::Texture(t)) => t.texture.create_view(&Default::default()),
        _ => return 0,
    };
    ht.alloc(GpuObject::TextureView(Box::new(view)))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_texture_write")]
pub fn gfx_texture_write(tex_h: i32, dev_h: i32, data: &[u8], bytes_per_row: i32, height: i32) {
    let ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return,
    };
    if let Some(GpuObject::Texture(t)) = ht.get(tex_h) {
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &t.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row as u32),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: t.width,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_texture_destroy")]
pub fn gfx_texture_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Sampler
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_sampler_create")]
pub fn gfx_sampler_create(dev_h: i32, min: i32, mag: i32, addr: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let f = |v: i32| {
        if v == 1 {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        }
    };
    let a = match addr {
        1 => wgpu::AddressMode::Repeat,
        2 => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    };
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        min_filter: f(min),
        mag_filter: f(mag),
        address_mode_u: a,
        address_mode_v: a,
        ..Default::default()
    });
    ht.alloc(GpuObject::Sampler(Box::new(GraphicsSampler { sampler })))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_sampler_destroy")]
pub fn gfx_sampler_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Pipeline
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_begin")]
pub fn gfx_pipeline_begin() -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    ht.alloc(GpuObject::PipelineBuilder(Box::new(PipelineBuilder::new())))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_shader")]
pub fn gfx_pipeline_set_shader(pipe_h: i32, shader_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(b)) = ht.get_mut(pipe_h) {
        b.shader = Some(shader_h);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_format")]
pub fn gfx_pipeline_set_format(pipe_h: i32, format: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(b)) = ht.get_mut(pipe_h) {
        let fmt = types::int_to_texture_format(format);
        if b.color_targets.is_empty() {
            b.color_targets.push(fmt);
        } else {
            b.color_targets[0] = fmt;
        }
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_topology")]
pub fn gfx_pipeline_set_topology(pipe_h: i32, topo: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(b)) = ht.get_mut(pipe_h) {
        b.topology = types::primitive_topology_from_int(topo);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_cull")]
pub fn gfx_pipeline_set_cull(pipe_h: i32, cull: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(b)) = ht.get_mut(pipe_h) {
        b.cull_mode = types::cull_mode_from_int(cull);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_add_color_target")]
pub fn gfx_pipeline_add_color_target(pipe_h: i32, format: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(b)) = ht.get_mut(pipe_h) {
        b.color_targets.push(types::int_to_texture_format(format));
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_build")]
pub fn gfx_pipeline_build(pipe_h: i32, dev_h: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let device = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c.device.clone(),
        _ => return 0,
    };

    // Get shader from handle
    let builder = match ht.remove(pipe_h) {
        Some(GpuObject::PipelineBuilder(b)) => b,
        _ => return 0,
    };

    let shader_h = match builder.shader {
        Some(h) => h,
        None => return 0,
    };
    let (module, vs_entry, fs_entry) = match ht.get(shader_h) {
        Some(GpuObject::Shader {
            module,
            vs_entry,
            fs_entry,
        }) => (module, vs_entry.clone(), fs_entry.clone()),
        _ => return 0,
    };

    let targets: Vec<Option<wgpu::ColorTargetState>> = if builder.color_targets.is_empty() {
        vec![Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Bgra8Unorm,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })]
    } else {
        builder
            .color_targets
            .iter()
            .map(|f| {
                Some(wgpu::ColorTargetState {
                    format: *f,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect()
    };

    let pipeline = match device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(&vs_entry),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(&fs_entry),
            targets: &targets,
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: builder.topology,
            cull_mode: builder.cull_mode,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    }) {
        p => p,
    };

    ht.alloc(GpuObject::Pipeline(Box::new(GraphicsPipeline { pipeline })))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_destroy")]
pub fn gfx_pipeline_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Command Encoder
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_create")]
pub fn gfx_cmd_create() -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    ht.alloc(GpuObject::CommandRecorder(Box::new(CommandRecorder::new())))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_submit")]
pub fn gfx_cmd_submit(cmd_h: i32, dev_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    // Clone Arc refs to avoid borrow conflict
    let (device, queue) = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => (c.device.clone(), c.queue.clone()),
        _ => return,
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        unsafe {
            cmd.submit(&device, &queue);
        }
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_destroy")]
pub fn gfx_cmd_destroy(h: i32) {
    HANDLES.lock().unwrap().free(h);
}

// ============================================================================
// Compute
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_compute_is_available")]
pub fn compute_is_available() -> i32 {
    gfx_is_available()
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_create")]
pub async fn compute_create() -> i32 {
    gfx_device_create().await
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_destroy")]
pub fn compute_destroy(h: i32) {
    gfx_device_destroy(h);
}
