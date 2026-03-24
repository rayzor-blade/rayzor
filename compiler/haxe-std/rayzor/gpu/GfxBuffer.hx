package rayzor.gpu;

/**
 * GPU buffer for vertex, index, uniform, or storage data.
 *
 * Example:
 * ```haxe
 * var buf = GfxBuffer.create(device, 1024, BufferUsage.VERTEX | BufferUsage.COPY_DST);
 * buf.destroy();
 * ```
 */
@:native("rayzor::gpu::GfxBuffer")
extern class GfxBuffer {
    /** Create a buffer with the given size (bytes) and usage flags. */
    @:native("rayzor_gpu_gfx_buffer_create")
    public static function create(device:GPUDevice, size:Int, usageFlags:Int):GfxBuffer;

    /** Write data to the buffer at the given byte offset. */
    @:native("rayzor_gpu_gfx_buffer_write")
    public function write(device:GPUDevice, offset:Int, data:Dynamic, dataLen:Int):Void;

    /** Destroy this buffer and free GPU memory. */
    @:native("rayzor_gpu_gfx_buffer_destroy")
    public function destroy():Void;
}
