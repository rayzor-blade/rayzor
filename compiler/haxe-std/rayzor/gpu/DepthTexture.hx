package rayzor.gpu;

/**
 * Depth texture for 3D rendering with depth testing.
 *
 * Convenience wrapper that creates a depth-format texture with
 * RENDER_ATTACHMENT usage for depth buffer operations.
 *
 * Example:
 * ```haxe
 * var depth = DepthTexture.create(device, 800, 600);
 * // Use depth.view in render pass depth attachment
 * ```
 */
class DepthTexture {
    public var texture:Texture;
    public var view:Dynamic;
    public var width:Int;
    public var height:Int;

    public function new(tex:Texture, w:Int, h:Int) {
        texture = tex;
        view = tex.getView();
        width = w;
        height = h;
    }

    /** Create a Depth32Float texture. */
    public static function create(device:GPUDevice, width:Int, height:Int):DepthTexture {
        // format=3 (Depth32Float), usage=16 (RENDER_ATTACHMENT)
        var tex = Texture.create(device, width, height, 3, 16);
        return new DepthTexture(tex, width, height);
    }

    /** Create a Depth24PlusStencil8 texture. */
    public static function createWithStencil(device:GPUDevice, width:Int, height:Int):DepthTexture {
        // format=2 (Depth24PlusStencil8), usage=16 (RENDER_ATTACHMENT)
        var tex = Texture.create(device, width, height, 2, 16);
        return new DepthTexture(tex, width, height);
    }

    public function destroy():Void {
        texture.destroy();
    }
}
