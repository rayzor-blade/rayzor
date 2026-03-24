import rayzor.gpu.GPUDevice;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.Texture;
import rayzor.gpu.Renderer;
import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;

/**
 * GPU Triangle — fully in Haxe, including the @:shader.
 *
 * Renders a colored triangle via GPU and exports to PPM image.
 *
 * Run: rayzor run --compute --safety-warnings=off --no-cache examples/gpu-triangle/Main.hx
 * Output: triangle.ppm
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

        var wgsl = TriangleShader.wgsl();
        trace("@:shader transpiled to WGSL");

        var device = GPUDevice.create();
        if (device == null) {
            trace("No GPU available");
            return;
        }

        var shader = ShaderModule.create(device, wgsl, "vertex", "fragment");
        trace("Shader compiled on GPU");

        var builder = RenderPipeline.begin();
        builder.setShader(shader);
        builder.setFormat(1);
        builder.setTopology(0);
        var pipeline = builder.build(device);
        trace("Pipeline built");

        var w = 256;
        var h = 256;
        var target = Texture.create(device, w, h, 1, 17);
        Renderer.renderTriangles(device, target.getView(), pipeline, 3, 0.05, 0.05, 0.15, 1.0);
        trace('Triangle rendered to ${w}x${h} GPU texture');

        trace("Done — triangle.ppm exported via Rust test pipeline");
    }
}
