package rayzor.gpu;

/**
 * Depth texture for 3D rendering with depth testing.
 *
 * Auto-released via @:derive([Drop]). No manual destroy() needed.
 */
@:derive([Drop])
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

    public static function create(device:GPUDevice, width:Int, height:Int):DepthTexture {
        var tex = Texture.create(device, width, height, 3, 16);
        return new DepthTexture(tex, width, height);
    }

    public static function createWithStencil(device:GPUDevice, width:Int, height:Int):DepthTexture {
        var tex = Texture.create(device, width, height, 2, 16);
        return new DepthTexture(tex, width, height);
    }

    /** Called automatically when this DepthTexture is dropped. */
    public function drop():Void {
        if (texture != null) texture.destroy();
    }
}
