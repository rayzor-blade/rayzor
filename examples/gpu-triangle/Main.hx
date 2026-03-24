import rayzor.gpu.GPUDevice;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.Texture;
import rayzor.gpu.Renderer;
import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;
import haxe.io.Bytes;

/**
 * GPU Triangle — fully in Haxe, including shader and image export.
 *
 * 1. @:shader transpiles Haxe → WGSL at compile time
 * 2. GPU renders colored triangle to texture
 * 3. Pixels read back to haxe.io.Bytes
 * 4. PPM image written with sys.io.File
 *
 * Run: rayzor run --compute --safety-warnings=off --no-cache examples/gpu-triangle/Main.hx
 * Output: triangle.ppm (256x256, open with any image viewer)
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

        // 1. Transpile @:shader to WGSL
        var wgsl = TriangleShader.wgsl();
        trace("@:shader → WGSL");

        // 2. GPU setup
        var device = GPUDevice.create();
        if (device == null) { trace("No GPU"); return; }

        var shader = ShaderModule.create(device, wgsl, "vertex", "fragment");
        var builder = RenderPipeline.begin();
        builder.setShader(shader);
        builder.setFormat(1);   // RGBA8Unorm
        builder.setTopology(0); // TriangleList
        var pipeline = builder.build(device);
        trace("GPU pipeline ready");

        // 3. Render to texture
        var w = 256;
        var h = 256;
        var target = Texture.create(device, w, h, 1, 17); // RENDER_ATTACHMENT | COPY_SRC
        Renderer.renderTriangles(device, target.getView(), pipeline, 3, 0.05, 0.05, 0.15, 1.0);
        trace('Rendered ${w}x${h}');

        // 4. Read pixels from GPU → haxe.io.Bytes
        var pixels = target.toBytes(device);
        trace('GPU readback: ${pixels.length} bytes');

        // 5. Write PPM image using pure Haxe stdlib
        // PPM P6 header: "P6\n256 256\n255\n" then raw RGB
        var nl = String.fromCharCode(10);
        var headerStr = "P6" + nl + Std.string(w) + " " + Std.string(h) + nl + "255" + nl;
        var hdrLen = headerStr.length;
        var rgbSize = w * h * 3;
        var ppm = Bytes.alloc(hdrLen + rgbSize);

        // Write header bytes manually (Bytes.ofString has a known issue)
        for (i in 0...hdrLen) {
            ppm.set(i, StringTools.fastCodeAt(headerStr, i));
        }

        // RGBA → RGB conversion
        var offset = hdrLen;
        for (i in 0...(w * h)) {
            ppm.set(offset, pixels.get(i * 4));         // R
            ppm.set(offset + 1, pixels.get(i * 4 + 1)); // G
            ppm.set(offset + 2, pixels.get(i * 4 + 2)); // B
            offset += 3;
        }

        sys.io.File.saveBytes("triangle.ppm", ppm);
        trace('Saved triangle.ppm (${ppm.length} bytes)');
        trace("Done");
    }
}
