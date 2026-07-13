package rayzor.gpu;

/**
 * Mesh — holds vertex buffer, optional index buffer, and vertex count.
 *
 * Resources are auto-released via @:derive([Drop]).
 */
@:derive([Drop])
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

    /** Called automatically when this Mesh is dropped. */
    public function drop():Void {
        if (vertexBuffer != null) vertexBuffer.destroy();
        if (indexBuffer != null) indexBuffer.destroy();
    }

    public static function fullscreenTriangle(device:GPUDevice):Mesh {
        var m = new Mesh();
        m.vertexCount = 3;
        m.stride = 12;
        return m;
    }

    public static function coloredTriangle(device:GPUDevice):Mesh {
        var m = new Mesh();
        m.vertexCount = 3;
        m.stride = 24;
        return m;
    }
}
