//! WASM host exports via wasm-bindgen.
//!
//! When the GPU crate is compiled with `--features wasm-host` and
//! `--target wasm32-unknown-unknown`, these functions are exported
//! as the JS host module for @:jsImport("rayzor-gpu").
//!
//! They wrap the same core GPU logic (wgpu) that the native extern "C"
//! functions use, but with wasm-bindgen marshaling instead of raw pointers.

use wasm_bindgen::prelude::*;

use crate::graphics::GraphicsContext;

// Global device state (matches native singleton pattern)
static mut GFX_CTX: Option<Box<GraphicsContext>> = None;

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_is_available")]
pub fn gfx_is_available() -> bool {
    // On WASM, WebGPU availability is checked by the JS init() call
    // before WASM starts. If we got here, device was initialized.
    unsafe { GFX_CTX.is_some() }
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_device_create")]
pub async fn gfx_device_create() -> i32 {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
        ..Default::default()
    });

    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
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
                required_features: wgpu::Features::empty(),
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

    unsafe {
        GFX_CTX = Some(Box::new(ctx));
    }

    1 // success handle
}

#[wasm_bindgen(js_name = "rayzor_gpu_gfx_device_destroy")]
pub fn gfx_device_destroy() {
    unsafe {
        GFX_CTX = None;
    }
}

// Additional exports will be added as needed.
// The wasm-pack build generates JS glue that bridges these to the
// @:jsImport("rayzor-gpu") functions the Haxe WASM module imports.
