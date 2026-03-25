package rayzor.gpu;

/**
 * Window surface for real-time frame presentation.
 *
 * Created from opaque window/display handles obtained from the OS.
 * Use TinyCC (`rayzor.runtime.CC`) or any native binding to request
 * a window from the system, then pass the handles here.
 *
 * Example:
 * ```haxe
 * // Get window handle from the system (via CC, native lib, etc.)
 * var handles = NativeWindow.create(1280, 720);
 * var surface = Surface.create(device, handles.window, handles.display, 1280, 720);
 *
 * // Render loop
 * var view = surface.getTexture();
 * cmd.beginPass(view, 0, 0.0, 0.0, 0.0, 1.0, null);
 * // ... draw ...
 * cmd.endPass();
 * cmd.submit(device);
 * surface.present();
 * ```
 */
@:native("rayzor::gpu::Surface")
extern class Surface {
    /** Create a surface from opaque window and display handles.
     *  The handles are platform-specific pointers obtained from the OS
     *  windowing system (via TinyCC, native libraries, etc.).
     */
    @:native("rayzor_gpu_gfx_surface_create")
    public static function create(device:GPUDevice, windowHandle:rayzor.Usize, displayHandle:rayzor.Usize, width:Int, height:Int):Surface;

    /** Get the current frame's texture view for rendering into. */
    @:native("rayzor_gpu_gfx_surface_get_texture")
    public function getTexture():TextureView;

    /** Present the rendered frame to the window. */
    @:native("rayzor_gpu_gfx_surface_present")
    public function present():Void;

    /** Resize the surface (call when window size changes). */
    @:native("rayzor_gpu_gfx_surface_resize")
    public function resize(device:GPUDevice, width:Int, height:Int):Void;

    /** Get the surface's preferred texture format (as TextureFormat int code). */
    @:native("rayzor_gpu_gfx_surface_get_format")
    public function getFormat():Int;

    /** Destroy this surface. */
    @:native("rayzor_gpu_gfx_surface_destroy")
    public function destroy():Void;
}
