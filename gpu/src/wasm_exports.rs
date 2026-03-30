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
use crate::wgpu_backend::buffer_ops::WgpuBuffer;

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
    ComputeBuffer(Box<ComputeBufferInfo>),
}

struct ComputeBufferInfo {
    buffer: WgpuBuffer,
    numel: u32,
    dtype: u8,
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
    // Two-phase: get texture from surface, then alloc view handle.
    // Can't hold mutable borrow of surface while allocating.
    let view = {
        let surf = match ht.get_mut(h) {
            Some(GpuObject::Surface(s)) => s,
            _ => return 0,
        };
        match surf.surface.get_current_texture() {
            Ok(frame) => {
                let v = frame.texture.create_view(&Default::default());
                surf.current_texture = Some(frame);
                Some(v)
            }
            Err(_) => None,
        }
    }; // surf borrow ends here
    match view {
        Some(v) => ht.alloc(GpuObject::TextureView(Box::new(v))),
        None => 0,
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

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_begin_pass")]
pub fn gfx_cmd_begin_pass(
    cmd_h: i32,
    color_view_h: i32,
    load_op: i32,
    clear_r: f64,
    clear_g: f64,
    clear_b: f64,
    clear_a: f64,
    depth_view_h: i32,
) {
    let mut ht = HANDLES.lock().unwrap();
    // Get raw pointers before mutable borrow of cmd
    let color_view_ptr = match ht.get(color_view_h) {
        Some(GpuObject::TextureView(v)) => &**v as *const wgpu::TextureView,
        _ => return,
    };
    let depth_view_ptr = if depth_view_h != 0 {
        match ht.get(depth_view_h) {
            Some(GpuObject::TextureView(v)) => &**v as *const wgpu::TextureView,
            _ => std::ptr::null(),
        }
    } else {
        std::ptr::null()
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.begin_pass(color_view_ptr, load_op, clear_r, clear_g, clear_b, clear_a, depth_view_ptr);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_pipeline")]
pub fn gfx_cmd_set_pipeline(cmd_h: i32, pipeline_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let pipeline_ptr = match ht.get(pipeline_h) {
        Some(GpuObject::Pipeline(p)) => &**p as *const GraphicsPipeline,
        _ => return,
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_pipeline(pipeline_ptr);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_vertex_buffer")]
pub fn gfx_cmd_set_vertex_buffer(cmd_h: i32, slot: u32, buffer_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let buffer_ptr = match ht.get(buffer_h) {
        Some(GpuObject::Buffer(b)) => &**b as *const crate::graphics::GraphicsBuffer,
        _ => return,
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_vertex_buffer(slot, buffer_ptr);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_index_buffer")]
pub fn gfx_cmd_set_index_buffer(cmd_h: i32, buffer_h: i32, format: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let buffer_ptr = match ht.get(buffer_h) {
        Some(GpuObject::Buffer(b)) => &**b as *const crate::graphics::GraphicsBuffer,
        _ => return,
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_index_buffer(buffer_ptr, format);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_bind_group")]
pub fn gfx_cmd_set_bind_group(cmd_h: i32, group_index: u32, bind_group_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let bg_ptr = match ht.get(bind_group_h) {
        Some(GpuObject::BindGroup(bg)) => &**bg as *const GraphicsBindGroup,
        _ => return,
    };
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_bind_group(group_index, bg_ptr);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_draw")]
pub fn gfx_cmd_draw(cmd_h: i32, vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_draw(vertex_count, instance_count, first_vertex, first_instance);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_draw_indexed")]
pub fn gfx_cmd_draw_indexed(cmd_h: i32, index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_draw_indexed(index_count, instance_count, first_index, base_vertex, first_instance);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_viewport")]
pub fn gfx_cmd_set_viewport(cmd_h: i32, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_viewport(x, y, w, h, min_depth, max_depth);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_set_scissor")]
pub fn gfx_cmd_set_scissor(cmd_h: i32, x: u32, y: u32, w: u32, h: u32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.push_set_scissor(x, y, w, h);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_begin_pass_mrt")]
pub fn gfx_cmd_begin_pass_mrt(cmd_h: i32, _count: i32, _color_views: &[i32], _load_ops: &[i32], _clear_colors: &[f64], _depth_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(_cmd)) = ht.get_mut(cmd_h) {
        // MRT stub — single pass fallback for now
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_cmd_end_pass")]
pub fn gfx_cmd_end_pass(cmd_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::CommandRecorder(cmd)) = ht.get_mut(cmd_h) {
        cmd.end_pass();
    }
}

// ============================================================================
// Pipeline extras
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_vertex_layout_simple")]
pub fn gfx_pipeline_set_vertex_layout_simple(builder_h: i32, stride: i32, attr_count: i32, attr_data: &[i32]) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(pb)) = ht.get_mut(builder_h) {
        pb.set_vertex_layout_simple(stride, attr_count, attr_data);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_set_depth_simple")]
pub fn gfx_pipeline_set_depth_simple(builder_h: i32, depth_format: i32) {
    let mut ht = HANDLES.lock().unwrap();
    if let Some(GpuObject::PipelineBuilder(pb)) = ht.get_mut(builder_h) {
        pb.set_depth_simple(depth_format);
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_pipeline_add_layout")]
pub fn gfx_pipeline_add_layout(builder_h: i32, layout_h: i32) {
    let mut ht = HANDLES.lock().unwrap();
    let layout_ptr = match ht.get(layout_h) {
        Some(GpuObject::BindGroupLayout(l)) => &**l as *const GraphicsBindGroupLayout,
        _ => return,
    };
    if let Some(GpuObject::PipelineBuilder(pb)) = ht.get_mut(builder_h) {
        pb.add_layout(layout_ptr);
    }
}

// ============================================================================
// Buffer extras
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_from_bytes")]
pub fn gfx_buffer_from_bytes(dev_h: i32, data: &[u8], usage_flags: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let usage = wgpu::BufferUsages::from_bits_truncate(usage_flags as u32);
    let buffer = wgpu::util::DeviceExt::create_buffer_init(
        &*ctx.device,
        &wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage,
        },
    );
    let gfx_buf = crate::graphics::GraphicsBuffer {
        buffer,
        size: data.len() as u64,
    };
    ht.alloc(GpuObject::Buffer(Box::new(gfx_buf)))
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_buffer_write_bytes")]
pub fn gfx_buffer_write_bytes(buf_h: i32, dev_h: i32, offset: i32, data: &[u8]) {
    let ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return,
    };
    if let Some(GpuObject::Buffer(b)) = ht.get(buf_h) {
        ctx.queue
            .write_buffer(&b.buffer, offset as u64, data);
    }
}

// ============================================================================
// Surface extras
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_surface_create")]
pub fn gfx_surface_create(
    dev_h: i32,
    _window_handle: i32,
    _display_handle: i32,
    width: i32,
    height: i32,
) -> i32 {
    // On WASM, ignore raw handles — find the existing canvas on the page.
    // Window.createCentered already created a canvas element.
    gfx_surface_create_canvas(dev_h, "", width, height)
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

// ============================================================================
// Compute buffer operations
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_compute_alloc_buffer")]
pub fn compute_alloc_buffer(dev_h: i32, numel: i32, dtype: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let elem_size: usize = match dtype {
        2 => 8,
        _ => 4,
    }; // F64=8, else 4
    let byte_size = (numel as usize) * elem_size;
    let wgpu_ctx = crate::wgpu_backend::device_init::WgpuContext {
        device: (*ctx.device).clone(),
        queue: (*ctx.queue).clone(),
    };
    match WgpuBuffer::allocate(&wgpu_ctx, byte_size) {
        Some(buf) => ht.alloc(GpuObject::ComputeBuffer(Box::new(ComputeBufferInfo {
            buffer: buf,
            numel: numel as u32,
            dtype: dtype as u8,
        }))),
        None => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_free_buffer")]
pub fn compute_free_buffer(_dev_h: i32, buf_h: i32) {
    HANDLES.lock().unwrap().free(buf_h);
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_buffer_numel")]
pub fn compute_buffer_numel(buf_h: i32) -> i32 {
    let ht = HANDLES.lock().unwrap();
    match ht.get(buf_h) {
        Some(GpuObject::ComputeBuffer(b)) => b.numel as i32,
        _ => 0,
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_buffer_dtype")]
pub fn compute_buffer_dtype(buf_h: i32) -> i32 {
    let ht = HANDLES.lock().unwrap();
    match ht.get(buf_h) {
        Some(GpuObject::ComputeBuffer(b)) => b.dtype as i32,
        _ => 0,
    }
}

// ============================================================================
// Compute elementwise ops — compile WGSL + dispatch
// ============================================================================

fn compute_binary_op(dev_h: i32, a_h: i32, b_h: i32, op: &str) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let (a_buf, a_numel) = match ht.get(a_h) {
        Some(GpuObject::ComputeBuffer(b)) => (&b.buffer.buffer, b.numel),
        _ => return 0,
    };
    let b_buf = match ht.get(b_h) {
        Some(GpuObject::ComputeBuffer(b)) => &b.buffer.buffer,
        _ => return 0,
    };

    let wgsl = format!(
        "@group(0) @binding(0) var<storage,read> a: array<f32>;\n\
         @group(0) @binding(1) var<storage,read> b: array<f32>;\n\
         @group(0) @binding(2) var<storage,read_write> out: array<f32>;\n\
         @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {{\n\
           let i = id.x;\n\
           if (i < arrayLength(&a)) {{ out[i] = a[i] {} b[i]; }}\n\
         }}",
        op
    );

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out_size = (a_numel as usize) * 4;
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((a_numel + 255) / 256) as u32, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let wgpu_buf = WgpuBuffer {
        buffer: out_buf,
        byte_size: out_size,
        device: &*ctx.device as *const _,
        queue: &*ctx.queue as *const _,
    };
    ht.alloc(GpuObject::ComputeBuffer(Box::new(ComputeBufferInfo {
        buffer: wgpu_buf,
        numel: a_numel,
        dtype: 1,
    })))
}

fn compute_unary_op(dev_h: i32, a_h: i32, expr: &str) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let (a_buf, a_numel) = match ht.get(a_h) {
        Some(GpuObject::ComputeBuffer(b)) => (&b.buffer.buffer, b.numel),
        _ => return 0,
    };

    let wgsl = format!(
        "@group(0) @binding(0) var<storage,read> a: array<f32>;\n\
         @group(0) @binding(1) var<storage,read_write> out: array<f32>;\n\
         @compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {{\n\
           let i = id.x;\n\
           if (i < arrayLength(&a)) {{ out[i] = {}; }}\n\
         }}",
        expr
    );

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out_size = (a_numel as usize) * 4;
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((a_numel + 255) / 256) as u32, 1, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let wgpu_buf = WgpuBuffer {
        buffer: out_buf,
        byte_size: out_size,
        device: &*ctx.device as *const _,
        queue: &*ctx.queue as *const _,
    };
    ht.alloc(GpuObject::ComputeBuffer(Box::new(ComputeBufferInfo {
        buffer: wgpu_buf,
        numel: a_numel,
        dtype: 1,
    })))
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_add")]
pub fn compute_add(dev: i32, a: i32, b: i32) -> i32 {
    compute_binary_op(dev, a, b, "+")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_sub")]
pub fn compute_sub(dev: i32, a: i32, b: i32) -> i32 {
    compute_binary_op(dev, a, b, "-")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_mul")]
pub fn compute_mul(dev: i32, a: i32, b: i32) -> i32 {
    compute_binary_op(dev, a, b, "*")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_div")]
pub fn compute_div(dev: i32, a: i32, b: i32) -> i32 {
    compute_binary_op(dev, a, b, "/")
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_neg")]
pub fn compute_neg(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "-a[i]")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_abs")]
pub fn compute_abs(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "abs(a[i])")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_sqrt")]
pub fn compute_sqrt(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "sqrt(a[i])")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_exp")]
pub fn compute_exp(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "exp(a[i])")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_log")]
pub fn compute_log(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "log(a[i])")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_relu")]
pub fn compute_relu(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "max(a[i], 0.0)")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_sigmoid")]
pub fn compute_sigmoid(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "1.0 / (1.0 + exp(-a[i]))")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_tanh")]
pub fn compute_tanh(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "tanh(a[i])")
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_gelu")]
pub fn compute_gelu(dev: i32, a: i32) -> i32 {
    compute_unary_op(
        dev,
        a,
        "a[i] * 0.5 * (1.0 + tanh(0.7978845608 * (a[i] + 0.044715 * a[i] * a[i] * a[i])))",
    )
}
#[wasm_bindgen(js_name = "rayzor_gpu_compute_silu")]
pub fn compute_silu(dev: i32, a: i32) -> i32 {
    compute_unary_op(dev, a, "a[i] / (1.0 + exp(-a[i]))")
}

// ============================================================================
// Compute reductions
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_compute_sum")]
pub fn compute_sum(dev_h: i32, buf_h: i32) -> f64 {
    compute_reduce(dev_h, buf_h, "s += a[i]", "0.0")
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_mean")]
pub fn compute_mean(dev_h: i32, buf_h: i32) -> f64 {
    let ht = HANDLES.lock().unwrap();
    let numel = match ht.get(buf_h) {
        Some(GpuObject::ComputeBuffer(b)) => b.numel as f64,
        _ => return 0.0,
    };
    drop(ht);
    let sum = compute_sum(dev_h, buf_h);
    if numel > 0.0 {
        sum / numel
    } else {
        0.0
    }
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_max")]
pub fn compute_max(dev_h: i32, buf_h: i32) -> f64 {
    compute_reduce(dev_h, buf_h, "s = max(s, a[i])", "-1e38")
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_min")]
pub fn compute_min(dev_h: i32, buf_h: i32) -> f64 {
    compute_reduce(dev_h, buf_h, "s = min(s, a[i])", "1e38")
}

fn compute_reduce(dev_h: i32, buf_h: i32, op: &str, init: &str) -> f64 {
    let ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0.0,
    };
    let (a_buf, _numel) = match ht.get(buf_h) {
        Some(GpuObject::ComputeBuffer(b)) => (&b.buffer.buffer, b.numel),
        _ => return 0.0,
    };

    let wgsl = format!(
        "@group(0) @binding(0) var<storage,read> a: array<f32>;\n\
         @group(0) @binding(1) var<storage,read_write> out: array<f32>;\n\
         @compute @workgroup_size(1) fn main() {{\n\
           var s: f32 = {init};\n\
           for (var i = 0u; i < arrayLength(&a); i++) {{ {op}; }}\n\
           out[0] = s;\n\
         }}"
    );

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 4);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    // Synchronous readback (works in WASM because wgpu handles it)
    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result = f32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f64;
    drop(data);
    read_buf.unmap();

    result
}

// ============================================================================
// Compute dot product + matmul
// ============================================================================

#[wasm_bindgen(js_name = "rayzor_gpu_compute_dot")]
pub fn compute_dot(dev_h: i32, a_h: i32, b_h: i32) -> f64 {
    let ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0.0,
    };
    let (a_buf, a_numel) = match ht.get(a_h) {
        Some(GpuObject::ComputeBuffer(b)) => (&b.buffer.buffer, b.numel),
        _ => return 0.0,
    };
    let b_buf = match ht.get(b_h) {
        Some(GpuObject::ComputeBuffer(b)) => &b.buffer.buffer,
        _ => return 0.0,
    };

    let wgsl = "@group(0) @binding(0) var<storage,read> a: array<f32>;\n\
         @group(0) @binding(1) var<storage,read> b: array<f32>;\n\
         @group(0) @binding(2) var<storage,read_write> out: array<f32>;\n\
         @compute @workgroup_size(1) fn main() {\n\
           var s: f32 = 0.0;\n\
           for (var i = 0u; i < arrayLength(&a); i++) { s += a[i] * b[i]; }\n\
           out[0] = s;\n\
         }";

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, 4);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result = f32::from_le_bytes([data[0], data[1], data[2], data[3]]) as f64;
    drop(data);
    read_buf.unmap();
    result
}

#[wasm_bindgen(js_name = "rayzor_gpu_compute_matmul")]
pub fn compute_matmul(dev_h: i32, a_h: i32, b_h: i32, m: i32, k: i32, n: i32) -> i32 {
    let mut ht = HANDLES.lock().unwrap();
    let ctx = match ht.get(dev_h) {
        Some(GpuObject::GfxContext(c)) => c,
        _ => return 0,
    };
    let a_buf = match ht.get(a_h) {
        Some(GpuObject::ComputeBuffer(b)) => &b.buffer.buffer,
        _ => return 0,
    };
    let b_buf = match ht.get(b_h) {
        Some(GpuObject::ComputeBuffer(b)) => &b.buffer.buffer,
        _ => return 0,
    };

    let wgsl = format!(
        "struct Params {{ m: u32, k: u32, n: u32 }}\n\
         @group(0) @binding(0) var<uniform> params: Params;\n\
         @group(0) @binding(1) var<storage,read> a: array<f32>;\n\
         @group(0) @binding(2) var<storage,read> b: array<f32>;\n\
         @group(0) @binding(3) var<storage,read_write> out: array<f32>;\n\
         @compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id: vec3u) {{\n\
           let row = id.x; let col = id.y;\n\
           if (row >= params.m || col >= params.n) {{ return; }}\n\
           var sum: f32 = 0.0;\n\
           for (var i = 0u; i < params.k; i++) {{\n\
             sum += a[row * params.k + i] * b[i * params.n + col];\n\
           }}\n\
           out[row * params.n + col] = sum;\n\
         }}"
    );

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let out_size = (m * n) as usize * 4;
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params_data = [m as u32, k as u32, n as u32];
    let params_bytes: &[u8] = bytemuck_cast_slice(&params_data);
    use wgpu::util::DeviceExt;
    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: params_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((m as u32 + 15) / 16, (n as u32 + 15) / 16, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let wgpu_buf = WgpuBuffer {
        buffer: out_buf,
        byte_size: out_size,
        device: &*ctx.device as *const _,
        queue: &*ctx.queue as *const _,
    };
    ht.alloc(GpuObject::ComputeBuffer(Box::new(ComputeBufferInfo {
        buffer: wgpu_buf,
        numel: (m * n) as u32,
        dtype: 1,
    })))
}

/// Safe cast for uniform buffer data
fn bytemuck_cast_slice(data: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
