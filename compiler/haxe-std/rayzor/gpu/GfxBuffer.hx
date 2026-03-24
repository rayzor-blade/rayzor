package rayzor.gpu;

/**
 * GPU buffer for vertex, index, uniform, or storage data.
 */
@:native("rayzor::gpu::GfxBuffer")
extern class GfxBuffer {
    /** Create an empty buffer with given size and usage flags. */
    @:native("rayzor_gpu_gfx_buffer_create")
    public static function create(device:GPUDevice, size:Int, usageFlags:Int):GfxBuffer;

    /** Create a buffer from haxe.io.Bytes data. */
    @:native("rayzor_gpu_gfx_buffer_from_bytes")
    public static function fromBytes(device:GPUDevice, data:haxe.io.Bytes, usageFlags:Int):GfxBuffer;

    /** Write haxe.io.Bytes to the buffer at an offset. */
    @:native("rayzor_gpu_gfx_buffer_write_bytes")
    public function writeBytes(device:GPUDevice, offset:Int, data:haxe.io.Bytes):Void;

    /** Destroy this buffer. */
    @:native("rayzor_gpu_gfx_buffer_destroy")
    public function destroy():Void;
}
