import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;

/**
 * GPU Triangle — fully written in Haxe, including the shader.
 *
 * Uses @:shader to transpile Haxe code to WGSL at compile time.
 * Saves a rasterized triangle as a PPM image file.
 *
 * Run: rayzor run --safety-warnings=off examples/gpu-triangle/Main.hx
 * Output: triangle.ppm (open with any image viewer)
 */

/** Inter-stage vertex shader output. */
@:gpuStruct
class VOut {
    public var position:Vec4;
    public var color:Vec3;
}

/** Triangle shader — Haxe code transpiled to WGSL at compile time. */
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
        trace("=== Rayzor GPU Triangle (pure Haxe) ===");

        // Transpile @:shader class to WGSL at compile time
        var wgsl = TriangleShader.wgsl();
        trace("--- Generated WGSL ---");
        trace(wgsl);
        trace("--- End WGSL ---");

        // Save CPU-rasterized triangle as PPM image
        var w = 256;
        var h = 256;
        var buf = new StringBuf();
        buf.add('P3\n${w} ${h}\n255\n');
        for (y in 0...h) {
            for (x in 0...w) {
                var fx = (x - w / 2.0) / (w / 2.0);
                var fy = (h / 2.0 - y) / (h / 2.0);
                if (inTri(fx, fy)) {
                    var c = bary(fx, fy);
                    buf.add('${Std.int(c.r * 255)} ${Std.int(c.g * 255)} ${Std.int(c.b * 255)} ');
                } else {
                    buf.add("13 13 38 ");
                }
            }
            buf.add("\n");
        }
        sys.io.File.saveContent("triangle.ppm", buf.toString());
        trace('Saved triangle.ppm (${w}x${h})');
        trace("Done");
    }

    static function inTri(px:Float, py:Float):Bool {
        var d1 = (px - 0.0) * ((-0.5) - 0.5) - ((-0.5) - 0.0) * (py - 0.5);
        var d2 = (px - (-0.5)) * ((-0.5) - (-0.5)) - (0.5 - (-0.5)) * (py - (-0.5));
        var d3 = (px - 0.5) * (0.5 - (-0.5)) - (0.0 - 0.5) * (py - (-0.5));
        var hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        var hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(hasNeg && hasPos);
    }

    static function bary(px:Float, py:Float):{r:Float, g:Float, b:Float} {
        var denom = ((-0.5) - (-0.5)) * (0.0 - 0.5) + (0.5 - (-0.5)) * (0.5 - (-0.5));
        if (Math.abs(denom) < 0.001) return {r: 0.33, g: 0.33, b: 0.34};
        var u = (((-0.5) - (-0.5)) * (px - 0.5) + (0.5 - (-0.5)) * (py - (-0.5))) / denom;
        var v = (((-0.5) - 0.5) * (px - 0.5) + (0.0 - 0.5) * (py - (-0.5))) / denom;
        var w = 1.0 - u - v;
        u = Math.max(0.0, Math.min(1.0, u));
        v = Math.max(0.0, Math.min(1.0, v));
        w = Math.max(0.0, Math.min(1.0, w));
        return {r: u, g: v, b: w};
    }
}
