package rayzor.gpu;

/**
 * GPU texture — 2D image for rendering or sampling.
 */
@:native("rayzor::gpu::Texture")
extern class Texture {
    /** Create a 2D texture with format and usage flags. */
    @:native("rayzor_gpu_gfx_texture_create")
    public static function create(device:GPUDevice, width:Int, height:Int, format:Int, usageFlags:Int):Texture;

    /** Create a Depth32Float texture for depth testing. */
    @:native("rayzor_gpu_gfx_depth_texture_create")
    public static function createDepth(device:GPUDevice, width:Int, height:Int):Texture;

    /** Get the default view for this texture. */
    @:native("rayzor_gpu_gfx_texture_get_view")
    public function getView():TextureView;

    /** Read pixel data from GPU to CPU as haxe.io.Bytes (RGBA8, 4 bytes/pixel). */
    @:native("rayzor_gpu_gfx_texture_to_bytes")
    public function toBytes(device:GPUDevice):haxe.io.Bytes;

    /** Upload pixel data from haxe.io.Bytes to texture. */
    @:native("rayzor_gpu_gfx_texture_upload")
    public function upload(device:GPUDevice, data:haxe.io.Bytes, bytesPerRow:Int):Void;

    /** Destroy this texture. */
    @:native("rayzor_gpu_gfx_texture_destroy")
    public function destroy():Void;
}
