import rayzor.gpu.GPUDevice;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.Texture;
import rayzor.gpu.Renderer;
import rayzor.gpu.TextureUsage;
import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;

/**
 * GPU Triangle — fully written in Haxe, including the shader.
 *
 * Uses @:shader to transpile Haxe code to WGSL at compile time.
 * Renders a colored triangle to a texture and saves as PPM image.
 *
 * Run: rayzor run examples/gpu-triangle/Main.hx
 */

// --- Shader (transpiled to WGSL at compile time) ---

@:gpuStruct
class VertexOutput {
    @:builtin("position") public var pos:Vec4;
    @:location(0) public var color:Vec3;
}

@:shader
class TriangleShader {
    @:vertex
    function vertex(@:builtin("vertex_index") idx:Int):VertexOutput {
        // Procedural triangle — positions and colors from vertex index
        var positions = [
            new Vec3(0.0, 0.5, 0.0),
            new Vec3(-0.5, -0.5, 0.0),
            new Vec3(0.5, -0.5, 0.0)
        ];
        var colors = [
            new Vec3(1.0, 0.0, 0.0),   // red
            new Vec3(0.0, 1.0, 0.0),   // green
            new Vec3(0.0, 0.0, 1.0)    // blue
        ];

        var out = new VertexOutput();
        out.pos = Vec4.fromVec3(positions[idx], 1.0);
        out.color = colors[idx];
        return out;
    }

    @:fragment
    function fragment(input:VertexOutput):Vec4 {
        return Vec4.fromVec3(input.color, 1.0);
    }
}

// --- Application ---

class Main {
    static function main() {
        trace("=== Rayzor GPU Triangle (pure Haxe) ===");

        // Check GPU availability
        if (!GPUDevice.isAvailable()) {
            trace("No GPU — generating CPU preview instead");
            saveCpuTriangle("triangle.ppm", 256, 256);
            return;
        }

        var device = GPUDevice.create();
        trace("GPU device created");

        // Compile shader from Haxe @:shader class — no WGSL strings!
        var wgslSource = TriangleShader.wgsl();
        trace("Shader transpiled from Haxe to WGSL:");
        trace(wgslSource);

        var shader = ShaderModule.create(device, wgslSource, "vertex", "fragment");
        trace("Shader compiled");

        // Build pipeline
        var builder = RenderPipeline.begin();
        builder.setShader(shader);
        builder.setFormat(1);      // RGBA8Unorm
        builder.setTopology(0);    // TriangleList
        var pipeline = builder.build(device);
        trace("Pipeline built");

        // Render to texture
        var w = 256;
        var h = 256;
        var target = Texture.create(device, w, h, 1,
            TextureUsage.RENDER_ATTACHMENT | TextureUsage.COPY_SRC);

        Renderer.submit(
            device, target.getView(),
            0,                          // Clear
            0.05, 0.05, 0.15, 1.0,     // dark navy
            null, pipeline,
            null, 3, 1,                 // 3 procedural vertices
            null, 0, 0, 0, null
        );
        trace('Rendered ${w}x${h} triangle to GPU texture');

        // Save as PPM (CPU-side rasterization to demonstrate file output)
        saveCpuTriangle("triangle.ppm", w, h);

        // Cleanup
        target.destroy();
        pipeline.destroy();
        shader.destroy();
        device.destroy();
        trace("Done");
    }

    /**
     * Save a colored triangle as a PPM image file.
     * PPM is a trivial format viewable by most image apps.
     * Uses CPU-side barycentric rasterization to match the GPU output.
     */
    static function saveCpuTriangle(path:String, w:Int, h:Int) {
        var buf = new StringBuf();
        buf.add('P3\n${w} ${h}\n255\n');

        for (y in 0...h) {
            for (x in 0...w) {
                // Normalized device coordinates
                var fx = (x - w / 2.0) / (w / 2.0);
                var fy = (h / 2.0 - y) / (h / 2.0);

                // Triangle: (0, 0.5), (-0.5, -0.5), (0.5, -0.5)
                if (inTriangle(fx, fy)) {
                    var b = bary(fx, fy);
                    buf.add('${Std.int(b.r * 255)} ${Std.int(b.g * 255)} ${Std.int(b.b * 255)} ');
                } else {
                    buf.add("13 13 38 "); // dark navy
                }
            }
            buf.add("\n");
        }

        sys.io.File.saveContent(path, buf.toString());
        trace('Saved ${path} (${w}x${h} PPM image)');
    }

    static function inTriangle(px:Float, py:Float):Bool {
        var d1 = edge(px, py, 0, 0.5, -0.5, -0.5);
        var d2 = edge(px, py, -0.5, -0.5, 0.5, -0.5);
        var d3 = edge(px, py, 0.5, -0.5, 0, 0.5);
        return !(((d1 < 0) || (d2 < 0) || (d3 < 0)) && ((d1 > 0) || (d2 > 0) || (d3 > 0)));
    }

    static function edge(x1:Float, y1:Float, x2:Float, y2:Float, x3:Float, y3:Float):Float {
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3);
    }

    static function bary(px:Float, py:Float):{r:Float, g:Float, b:Float} {
        var denom = (-0.5 - (-0.5)) * (0.0 - 0.5) + (0.5 - (-0.5)) * (0.5 - (-0.5));
        if (Math.abs(denom) < 0.001) return {r: 0.33, g: 0.33, b: 0.34};
        var u = ((-0.5 - (-0.5)) * (px - 0.5) + (0.5 - (-0.5)) * (py - (-0.5))) / denom;
        var v = (((-0.5) - 0.5) * (px - 0.5) + (0.0 - 0.5) * (py - (-0.5))) / denom;
        var w = 1.0 - u - v;
        u = Math.max(0, Math.min(1, u));
        v = Math.max(0, Math.min(1, v));
        w = Math.max(0, Math.min(1, w));
        return {r: u, g: v, b: w};
    }
}
