package rayzor.gpu;

/**
 * GPU texture — 2D image for rendering or sampling.
 *
 * Example:
 * ```haxe
 * var tex = Texture.create(device, 800, 600, TextureFormat.BGRA8Unorm,
 *     TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC);
 * var view = tex.getView();
 * tex.destroy();
 * ```
 */
@:native("rayzor::gpu::Texture")
extern class Texture {
    /** Create a 2D texture. */
    @:native("rayzor_gpu_gfx_texture_create")
    public static function create(device:GPUDevice, width:Int, height:Int, format:Int, usageFlags:Int):Texture;

    /** Get the default view for this texture. */
    @:native("rayzor_gpu_gfx_texture_get_view")
    public function getView():Dynamic;

    /** Read pixel data back from GPU to CPU. Returns RGBA8 bytes (4 bytes per pixel). */
    @:native("rayzor_gpu_gfx_texture_read_pixels")
    public function readPixels(device:GPUDevice, outPtr:Dynamic, outLen:Int):Int;

    /** Destroy this texture. */
    @:native("rayzor_gpu_gfx_texture_destroy")
    public function destroy():Void;
}
