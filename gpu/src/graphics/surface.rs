//! Surface creation from raw window handles.
//!
//! Accepts platform-native window pointers (NSView*, HWND, X11 Window)
//! and creates a wgpu Surface for real-time frame presentation.

use super::GraphicsContext;
use std::num::NonZeroU32;

pub struct GraphicsSurface {
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
    pub current_texture: Option<wgpu::SurfaceTexture>,
}

// ============================================================================
// Platform handle wrappers for raw-window-handle 0.6
// ============================================================================

/// Wraps raw platform pointers into the traits wgpu::Surface needs.
struct RawSurfaceTarget {
    window: raw_window_handle::RawWindowHandle,
    display: raw_window_handle::RawDisplayHandle,
}

// SAFETY: The raw handles are platform window pointers that are valid for the
// surface's lifetime. They're only used once during surface creation.
unsafe impl Send for RawSurfaceTarget {}
unsafe impl Sync for RawSurfaceTarget {}

impl raw_window_handle::HasWindowHandle for RawSurfaceTarget {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        // SAFETY: caller guarantees the handle is valid for the surface's lifetime
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(self.window) })
    }
}

impl raw_window_handle::HasDisplayHandle for RawSurfaceTarget {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Ok(unsafe { raw_window_handle::DisplayHandle::borrow_raw(self.display) })
    }
}

/// Build platform-specific raw handles from opaque pointers.
///
/// - macOS: window_handle = NSView*, display_handle = ignored
/// - Linux/X11: window_handle = X11 Window (u32 cast to ptr), display_handle = Display*
/// - Windows: window_handle = HWND, display_handle = HINSTANCE (or null)
fn make_raw_handles(
    window_handle: *mut std::ffi::c_void,
    display_handle: *mut std::ffi::c_void,
) -> Option<RawSurfaceTarget> {
    use raw_window_handle::*;

    #[cfg(target_os = "macos")]
    {
        if window_handle.is_null() {
            return None;
        }
        let ns_view = std::ptr::NonNull::new(window_handle)?;
        let window = RawWindowHandle::AppKit(AppKitWindowHandle::new(ns_view));
        let display = RawDisplayHandle::AppKit(AppKitDisplayHandle::new());
        Some(RawSurfaceTarget { window, display })
    }

    #[cfg(target_os = "linux")]
    {
        if window_handle.is_null() || display_handle.is_null() {
            return None;
        }
        // X11: window_handle is the Window ID (u32), display_handle is Display*
        let x11_window = window_handle as u32;
        let x11_display = std::ptr::NonNull::new(display_handle)?;
        let mut wh = XlibWindowHandle::new(x11_window as u64);
        let window = RawWindowHandle::Xlib(wh);
        let display = RawDisplayHandle::Xlib(XlibDisplayHandle::new(Some(x11_display), 0));
        Some(RawSurfaceTarget { window, display })
    }

    #[cfg(target_os = "windows")]
    {
        if window_handle.is_null() {
            return None;
        }
        let hwnd = NonZeroIsize::new(window_handle as isize)?;
        let window = RawWindowHandle::Win32(Win32WindowHandle::new(hwnd));
        let hinstance = NonZeroIsize::new(display_handle as isize);
        let mut dh = WindowsDisplayHandle::new();
        let display = RawDisplayHandle::Windows(dh);
        Some(RawSurfaceTarget { window, display })
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        let _ = (window_handle, display_handle);
        None
    }
}

// ============================================================================
// Extern "C" entry points
// ============================================================================

/// Create a surface from raw window + display handles.
///
/// `window_handle` and `display_handle` are platform-specific raw pointers:
/// - macOS: window_handle = NSView*, display_handle = ignored (pass null)
/// - Linux: window_handle = X11 Window, display_handle = X11 Display*
/// - Windows: window_handle = HWND, display_handle = HINSTANCE (or null)
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_create(
    ctx: *mut GraphicsContext,
    window_handle: *mut std::ffi::c_void,
    display_handle: *mut std::ffi::c_void,
    width: u32,
    height: u32,
) -> *mut GraphicsSurface {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let ctx = &*ctx;

    let target = match make_raw_handles(window_handle, display_handle) {
        Some(t) => t,
        None => {
            eprintln!("[GPU] Invalid window/display handle for surface creation");
            return std::ptr::null_mut();
        }
    };

    let surface = match ctx.instance.create_surface(target) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[GPU] Failed to create surface: {}", e);
            return std::ptr::null_mut();
        }
    };

    // Pick the preferred format or fall back to Bgra8Unorm
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
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::Fifo, // vsync
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&ctx.device, &config);

    Box::into_raw(Box::new(GraphicsSurface {
        surface,
        config,
        format,
        current_texture: None,
    }))
}

/// Resize an existing surface.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_resize(
    surface: *mut GraphicsSurface,
    ctx: *mut GraphicsContext,
    width: u32,
    height: u32,
) {
    if surface.is_null() || ctx.is_null() {
        return;
    }
    let surface = &mut *surface;
    let ctx = &*ctx;
    surface.config.width = width.max(1);
    surface.config.height = height.max(1);
    surface.surface.configure(&ctx.device, &surface.config);
}

/// Get the current frame texture view for rendering.
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

/// Get the surface's preferred texture format (as int code).
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_get_format(
    surface: *mut GraphicsSurface,
) -> i32 {
    if surface.is_null() {
        return 0;
    }
    let surface = &*surface;
    super::types::texture_format_to_int(surface.format)
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

/// Destroy the surface.
#[no_mangle]
pub unsafe extern "C" fn rayzor_gpu_gfx_surface_destroy(surface: *mut GraphicsSurface) {
    if !surface.is_null() {
        drop(Box::from_raw(surface));
    }
}
