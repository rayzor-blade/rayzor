package rayzor.gpu;

/**
 * Uniform buffer for passing data to shaders.
 *
 * Auto-released via @:derive([Drop]). No manual destroy() needed.
 */
@:derive([Drop])
class UniformBuffer {
    public var buffer:GfxBuffer;
    public var size:Int;

    public function new(buf:GfxBuffer, sz:Int) {
        buffer = buf;
        size = sz;
    }

    public static function create(device:GPUDevice, size:Int):UniformBuffer {
        var buf = GfxBuffer.create(device, size, BufferUsage.UNIFORM | BufferUsage.COPY_DST);
        return new UniformBuffer(buf, size);
    }

    /** Called automatically when this UniformBuffer is dropped. */
    public function drop():Void {
        if (buffer != null) buffer.destroy();
    }
}
