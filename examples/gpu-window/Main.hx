import rayzor.window.Window;
import rayzor.window.Key;
import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;
import rayzor.gpu.Vec4;

/**
 * GPU Window Rendering — renders a triangle to a native window.
 *
 * Shaders are written in Haxe and transpiled to WGSL at compile time
 * via @:shader. Uses rayzor-window and rayzor-gpu rpkgs.
 *
 * Run:
 *   rayzor run --rpkg rayzor-window.rpkg --rpkg rayzor-gpu.rpkg examples/gpu-window/Main.hx
 */

@:shader
class WindowShader {
    function vertex(vertexIndex:Int):Vec4 {
        var px = 0.0;
        var py = 0.0;
        if (vertexIndex == 0) { px = 0.0; py = 0.5; }
        if (vertexIndex == 1) { px = -0.5; py = -0.5; }
        if (vertexIndex == 2) { px = 0.5; py = -0.5; }
        return new Vec4(px, py, 0.0, 1.0);
    }

    function fragment():Vec4 {
        return new Vec4(1.0, 0.4, 0.1, 1.0);
    }
}

class Main {
    static function main() {
        // Window
        var win = Window.createCentered("Rayzor", 800, 600);
        trace("Window: " + win.getWidth() + "x" + win.getHeight());

        // GPU
        var device = GPUDevice.create();
        var surface = Surface.create(device, win.getHandle(), win.getDisplayHandle(), 800, 600);

        // Shader + Pipeline (Haxe → WGSL at compile time)
        var wgsl = WindowShader.wgsl();
        var shader = ShaderModule.create(device, wgsl, "vertex", "fragment");
        var pipe = RenderPipeline.begin();
        pipe.setShader(shader);
        pipe.setFormat(surface.getFormat());
        var built = pipe.build(device);

        // Render loop
        var frames = 0;
        while (win.pollEvents()) {
            if (win.isKeyDown(Key.ESCAPE)) break;

            var view = surface.getTexture();
            if (view == null) continue;

            var cmd = CommandEncoder.create();
            cmd.beginPass(view, 0, 0.05, 0.05, 0.08, 1.0, null);
            cmd.setPipeline(built);
            cmd.draw(3, 1, 0, 0);
            cmd.endPass();
            cmd.submit(device);
            surface.present();
            frames++;
        }

        trace("Rendered " + frames + " frames");
        built.destroy();
        shader.destroy();
        surface.destroy();
        device.destroy();
        win.destroy();
    }
}
