package rayzor.gpu;

/**
 * GPU surface for real-time frame presentation.
 *
 * On native: created from raw OS window handles (via Window.getHandle()).
 * On WASM: created from an HTML `<canvas>` element ID.
 *
 * Example (native):
 * ```haxe
 * var surface = Surface.create(device, win.getHandle(), win.getDisplayHandle(), 1280, 720);
 * ```
 *
 * Example (WASM):
 * ```haxe
 * var surface = Surface.createCanvas(device, "gpu-canvas", 1280, 720);
 * ```
 */
#if wasm
@:jsImport("rayzor-gpu")
#else
@:native("rayzor::gpu::Surface")
#end
extern class Surface {
    #if wasm
    @:jsMethod("create-surface-canvas")
    public static function createCanvas(device:GPUDevice, canvasId:String, width:Int, height:Int):Surface;

    @:jsMethod("create-surface")
    public static function create(device:GPUDevice, windowHandle:rayzor.Ptr<Void>, displayHandle:rayzor.Ptr<Void>, width:Int, height:Int):Surface;
    #else
    /** Create a surface from opaque window and display handles. */
    @:native("rayzor_gpu_gfx_surface_create")
    public static function create(device:GPUDevice, windowHandle:rayzor.Ptr<Void>, displayHandle:rayzor.Ptr<Void>, width:Int, height:Int):Surface;

    /** Create a surface from an HTML canvas element (WASM only — no-op on native). */
    @:native("rayzor_gpu_gfx_surface_create_canvas")
    public static function createCanvas(device:GPUDevice, canvasId:String, width:Int, height:Int):Surface;
    #end

    // Common methods — same on all platforms

    #if wasm
    @:jsMethod("surface-get-texture")
    public function getTexture():TextureView;
    @:jsMethod("surface-present")
    public function present():Void;
    @:jsMethod("surface-resize")
    public function resize(device:GPUDevice, width:Int, height:Int):Void;
    @:jsMethod("surface-get-format")
    public function getFormat():Int;
    @:jsMethod("surface-destroy")
    public function destroy():Void;
    #else
    @:native("rayzor_gpu_gfx_surface_get_texture")
    public function getTexture():TextureView;
    @:native("rayzor_gpu_gfx_surface_present")
    public function present():Void;
    @:native("rayzor_gpu_gfx_surface_resize")
    public function resize(device:GPUDevice, width:Int, height:Int):Void;
    @:native("rayzor_gpu_gfx_surface_get_format")
    public function getFormat():Int;
    @:native("rayzor_gpu_gfx_surface_destroy")
    public function destroy():Void;
    #end
}
