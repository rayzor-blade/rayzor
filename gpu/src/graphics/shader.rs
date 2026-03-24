//! WGSL shader module compilation.

use super::GraphicsContext;

pub struct GraphicsShader {
    pub module: wgpu::ShaderModule,
    pub vertex_entry: String,
    pub fragment_entry: String,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_shader_create(
    ctx: *mut GraphicsContext,
    wgsl_ptr: *const u8,
    wgsl_len: usize,
    vert_ptr: *const u8,
    vert_len: usize,
    frag_ptr: *const u8,
    frag_len: usize,
) -> *mut GraphicsShader {
    if ctx.is_null() || wgsl_ptr.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;
    let wgsl = std::str::from_utf8(std::slice::from_raw_parts(wgsl_ptr, wgsl_len)).unwrap_or("");
    let vert = std::str::from_utf8(std::slice::from_raw_parts(vert_ptr, vert_len)).unwrap_or("vs_main");
    let frag = std::str::from_utf8(std::slice::from_raw_parts(frag_ptr, frag_len)).unwrap_or("fs_main");

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rayzor_shader"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    Box::into_raw(Box::new(GraphicsShader {
        module,
        vertex_entry: vert.to_string(),
        fragment_entry: frag.to_string(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_shader_destroy(shader: *mut GraphicsShader) {
    if !shader.is_null() {
        drop(Box::from_raw(shader));
    }
}
