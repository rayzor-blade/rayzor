import rayzor.window.Window;
import rayzor.window.Key;
import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;

/**
 * GPU Window Rendering — renders a triangle to a native window.
 *
 * Uses rayzor-window rpkg for native windowing and rayzor-gpu rpkg
 * for GPU rendering. No TCC, no inline C — pure Haxe.
 *
 * Run:
 *   rayzor run --rpkg rayzor-window.rpkg --rpkg rayzor-gpu.rpkg examples/gpu-window/Main.hx
 */
class Main {
    static function main() {
        // Window
        var win = Window.createCentered("Rayzor", 800, 600);
        trace("Window: " + win.getWidth() + "x" + win.getHeight());

        // GPU
        var device = GPUDevice.create();
        var surface = Surface.create(device, win.getHandle(), win.getDisplayHandle(), 800, 600);

        // Shader + Pipeline
        var wgsl = "@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f { var pos = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5)); return vec4f(pos[vi], 0.0, 1.0); } @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.4, 0.1, 1.0); }";
        var shader = ShaderModule.create(device, wgsl, "vs", "fs");
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
