import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;
import rayzor.window.Window;

/**
 * GPU Window — real-time colored triangle.
 *
 * Same Haxe code works on both native and WASM.
 *
 * Native:
 *   rayzor run --rpkg rayzor-gpu.rpkg --rpkg rayzor-window.rpkg examples/gpu-window/Main.hx
 *
 * WASM (browser):
 *   cd examples/gpu-window && ./build-wasm.sh
 */
class Main {
    static function main() {
        trace("=== GPU Window Demo ===");

        if (!GPUDevice.isAvailable()) {
            trace("GPU not available");
            return;
        }

        var device = GPUDevice.create();
        var win = Window.createCentered("Rayzor GPU", 800, 600);

        var surface = Surface.create(device, win.getHandle(), win.getDisplayHandle(), 800, 600);

        var wgsl = "
struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) color: vec3f,
}
@vertex fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
    var colors = array<vec3f, 3>(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0));
    var out: VertexOutput;
    out.pos = vec4f(positions[idx], 0.0, 1.0);
    out.color = colors[idx];
    return out;
}
@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(in.color, 1.0);
}
";

        var shader = ShaderModule.create(device, wgsl, "vs_main", "fs_main");
        var pipe = RenderPipeline.begin();
        pipe.setShader(shader);
        pipe.setFormat(surface.getFormat());
        var pipeline = pipe.build(device);

        var cmd = CommandEncoder.create();
        var frames = 0;

        // runLoop: blocking on native, async (requestAnimationFrame) on WASM.
        // Cleanup must NOT run after runLoop on WASM — resources are still in use.
        trace("device=" + device + " surface=" + surface + " pipeline=" + pipeline + " cmd=" + cmd);

        Window.runLoop(win, function():Bool {
            var view = surface.getTexture();
            if (view == null) return true;

            cmd.beginPass(view, 0, 0.05, 0.05, 0.15, 1.0, null);
            cmd.setPipeline(pipeline);
            cmd.draw(3, 1, 0, 0);
            cmd.endPass();
            cmd.submit(device);
            surface.present();

            frames++;
            if (frames == 1) trace("First frame: view=" + view);
            if (frames % 120 == 0) trace("Frame " + frames);
            return true;
        });
    }
}
