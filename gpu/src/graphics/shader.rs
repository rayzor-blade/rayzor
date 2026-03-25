//! WGSL shader module compilation.

use super::GraphicsContext;
use rayzor_runtime::haxe_string::HaxeString;

pub struct GraphicsShader {
    pub module: wgpu::ShaderModule,
    pub vertex_entry: String,
    pub fragment_entry: String,
}

/// Helper: read a HaxeString* to a &str
unsafe fn hs_to_str<'a>(hs: *const HaxeString) -> &'a str {
    if hs.is_null() || (*hs).ptr.is_null() || (*hs).len == 0 {
        return "";
    }
    std::str::from_utf8(std::slice::from_raw_parts((*hs).ptr, (*hs).len)).unwrap_or("")
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

/// Low-level shader creation (ptr+len pairs, used by Rust tests)
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
    let vert =
        std::str::from_utf8(std::slice::from_raw_parts(vert_ptr, vert_len)).unwrap_or("vs_main");
    let frag =
        std::str::from_utf8(std::slice::from_raw_parts(frag_ptr, frag_len)).unwrap_or("fs_main");

    create_shader_impl(ctx, wgsl, vert, frag)
}

/// Haxe-friendly shader creation (accepts HaxeString* directly)
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_shader_create_hx(
    ctx: *mut GraphicsContext,
    wgsl: *const HaxeString,
    vert_entry: *const HaxeString,
    frag_entry: *const HaxeString,
) -> *mut GraphicsShader {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;
    let wgsl_str = hs_to_str(wgsl);
    let vert_str = hs_to_str(vert_entry);
    let frag_str = hs_to_str(frag_entry);
    let vert = if vert_str.is_empty() {
        "vs_main"
    } else {
        vert_str
    };
    let frag = if frag_str.is_empty() {
        "fs_main"
    } else {
        frag_str
    };

    create_shader_impl(ctx, wgsl_str, vert, frag)
}

fn create_shader_impl(
    ctx: &GraphicsContext,
    wgsl: &str,
    vert: &str,
    frag: &str,
) -> *mut GraphicsShader {
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
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
