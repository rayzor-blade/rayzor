import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;

/**
 * GPU Triangle — fully in Haxe, including the shader.
 *
 * @:shader transpiles Haxe → WGSL at compile time.
 *
 * Run: rayzor run --safety-warnings=off examples/gpu-triangle/Main.hx
 */

@:gpuStruct
class VOut {
    public var position:Vec4;
    public var color:Vec3;
}

@:shader
class TriangleShader {
    function vertex(vertexIndex:Int):VOut {
        var px = 0.0;
        var py = 0.0;
        var r = 0.0;
        var g = 0.0;
        var b = 0.0;
        if (vertexIndex == 0) { px = 0.0; py = 0.5; r = 1.0; }
        if (vertexIndex == 1) { px = -0.5; py = -0.5; g = 1.0; }
        if (vertexIndex == 2) { px = 0.5; py = -0.5; b = 1.0; }

        var out = new VOut();
        out.position = new Vec4(px, py, 0.0, 1.0);
        out.color = new Vec3(r, g, b);
        return out;
    }

    function fragment(input:VOut):Vec4 {
        return new Vec4(input.color.x, input.color.y, input.color.z, 1.0);
    }
}

class Main {
    static function main() {
        trace("=== Rayzor GPU Triangle ===");
        trace("");
        trace("--- Generated WGSL from @:shader ---");
        trace(TriangleShader.wgsl());
        trace("--- End WGSL ---");
        trace("Done");
    }
}
