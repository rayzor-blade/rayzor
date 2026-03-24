//! Surface creation from raw window handles.

use super::types::texture_format_from_int;
use super::GraphicsContext;

pub struct GraphicsSurface {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    pub current_texture: Option<wgpu::SurfaceTexture>,
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

/// Create a surface from raw window + display handles.
/// `window_handle` and `display_handle` are platform-specific raw pointers.
/// On macOS: window_handle = NSView*, display_handle = ignored (can be null).
/// On Linux: window_handle = X11 Window, display_handle = X11 Display*.
/// On Windows: window_handle = HWND, display_handle = HINSTANCE.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_create(
    ctx: *mut GraphicsContext,
    _window_handle: *mut std::ffi::c_void,
    _display_handle: *mut std::ffi::c_void,
    width: u32,
    height: u32,
) -> *mut GraphicsSurface {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let _ctx = &*ctx;

    // TODO: Create surface from raw window handle using raw-window-handle crate.
    // For now, return null — headless rendering uses textures directly.
    // Full windowed rendering requires platform-specific handle conversion.
    eprintln!("[GPU] Surface creation from raw handles not yet implemented — use headless render-to-texture");
    let _ = width;
    let _ = height;
    std::ptr::null_mut()
}

/// Resize an existing surface.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_resize(
    surface: *mut GraphicsSurface,
    width: u32,
    height: u32,
) {
    if surface.is_null() {
        return;
    }
    let surface = &mut *surface;
    surface.config.width = width;
    surface.config.height = height;
}

/// Get the current frame texture for rendering.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_get_texture(
    surface: *mut GraphicsSurface,
) -> *mut wgpu::TextureView {
    if surface.is_null() {
        return std::ptr::null_mut();
    }
    let surface = &mut *surface;

    let frame = match surface.surface.get_current_texture() {
        Ok(f) => f,
        Err(_) => return std::ptr::null_mut(),
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    surface.current_texture = Some(frame);
    Box::into_raw(Box::new(view))
}

/// Present the current frame.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_present(surface: *mut GraphicsSurface) {
    if surface.is_null() {
        return;
    }
    let surface = &mut *surface;
    if let Some(texture) = surface.current_texture.take() {
        texture.present();
    }
}

#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_destroy(surface: *mut GraphicsSurface) {
    if !surface.is_null() {
        drop(Box::from_raw(surface));
    }
}
