package rayzor.gpu;

/**
 * Mesh — holds vertex buffer, optional index buffer, and vertex count.
 *
 * Provides static factory methods for common shapes.
 *
 * Example:
 * ```haxe
 * var tri = Mesh.triangle(device);
 * var quad = Mesh.quad(device);
 * ```
 */
class Mesh {
    public var vertexBuffer:GfxBuffer;
    public var indexBuffer:GfxBuffer;
    public var vertexCount:Int;
    public var indexCount:Int;
    public var stride:Int;

    public function new() {
        vertexBuffer = null;
        indexBuffer = null;
        vertexCount = 0;
        indexCount = 0;
        stride = 0;
    }

    public function destroy():Void {
        if (vertexBuffer != null) vertexBuffer.destroy();
        if (indexBuffer != null) indexBuffer.destroy();
    }

    /** Create a full-screen triangle (3 vertices, position only). Covers the entire NDC. */
    public static function fullscreenTriangle(device:GPUDevice):Mesh {
        var m = new Mesh();
        m.vertexCount = 3;
        m.stride = 12; // 3 floats
        // Vertices defined in the shader via vertex_index — no buffer needed
        return m;
    }

    /** Create a colored triangle (3 vertices: position + color, 24 bytes/vertex). */
    public static function coloredTriangle(device:GPUDevice):Mesh {
        var m = new Mesh();
        m.vertexCount = 3;
        m.stride = 24; // 6 floats: xyz + rgb
        return m;
    }
}
