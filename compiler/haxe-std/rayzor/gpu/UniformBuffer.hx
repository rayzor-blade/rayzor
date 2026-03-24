package rayzor.gpu;

/**
 * Uniform buffer for passing data to shaders (camera matrices, colors, etc.).
 *
 * Wraps a GPU buffer with UNIFORM usage and provides typed write helpers.
 *
 * Example:
 * ```haxe
 * var ubo = UniformBuffer.create(device, 64); // 64 bytes (e.g., 4x4 matrix)
 * ubo.writeFloat(device, 0, 1.0);  // Write float at offset 0
 * ubo.writeVec4(device, 0, 1.0, 0.0, 0.0, 1.0); // RGBA color
 * ```
 */
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

    public function destroy():Void {
        buffer.destroy();
    }
}
