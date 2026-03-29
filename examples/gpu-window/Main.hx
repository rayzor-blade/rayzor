/**
 * GPU Window — real-time colored triangle.
 *
 * Works on both native and WASM. Same rendering code for both targets.
 *
 * Native:
 *   rayzor run --rpkg rayzor-gpu.rpkg --rpkg rayzor-window.rpkg examples/gpu-window/Main.hx
 *
 * WASM (browser):
 *   rayzor build --target wasm --browser examples/gpu-window/Main.hx
 */

// On native, these come from rpkg. On WASM, declare inline.
#if wasm
@:jsImport("rayzor-window")
extern class Window {
    @:jsMethod("create-centered")
    public static function createCentered(title:String, width:Int, height:Int):Window;
    @:jsMethod("poll-events")
    public function pollEvents():Bool;
    @:jsMethod("is-key-down")
    public function isKeyDown(keyCode:Int):Bool;
    @:jsMethod("get-handle")
    public function getHandle():Int;
    @:jsMethod("get-display-handle")
    public function getDisplayHandle():Int;
    @:jsMethod("get-width")
    public function getWidth():Int;
    @:jsMethod("get-height")
    public function getHeight():Int;
    @:jsMethod("was-resized")
    public function wasResized():Bool;
    @:jsMethod("run-loop")
    public static function runLoop(win:Window, callback:Dynamic):Void;
    @:jsMethod("destroy")
    public function destroy():Void;
}

@:jsImport("rayzor-gpu")
extern class GPUDevice {
    @:jsMethod("create-device")
    public static function create():GPUDevice;
    @:jsMethod("is-available")
    public static function isAvailable():Bool;
    @:jsMethod("destroy-device")
    public function destroy():Void;
}

@:jsImport("rayzor-gpu")
extern class GfxSurface {
    @:jsMethod("create-surface-canvas")
    public static function createCanvas(device:GPUDevice, canvasId:String, width:Int, height:Int):GfxSurface;
    @:jsMethod("surface-get-texture")
    public function getTexture():Int;
    @:jsMethod("surface-present")
    public function present():Void;
    @:jsMethod("surface-resize")
    public function resize(device:GPUDevice, width:Int, height:Int):Void;
    @:jsMethod("surface-get-format")
    public function getFormat():Int;
    @:jsMethod("surface-destroy")
    public function destroy():Void;
}

@:jsImport("rayzor-gpu")
extern class Shader {
    @:jsMethod("create-shader")
    public static function create(device:GPUDevice, wgsl:String, vs:String, fs:String):Shader;
    @:jsMethod("destroy-shader")
    public function destroy():Void;
}

@:jsImport("rayzor-gpu")
extern class Pipeline {
    @:jsMethod("pipeline-begin")
    public static function begin():Pipeline;
    @:jsMethod("pipeline-set-shader")
    public function setShader(shader:Shader):Void;
    @:jsMethod("pipeline-set-format")
    public function setFormat(format:Int):Void;
    @:jsMethod("pipeline-build")
    public function build(device:GPUDevice):Pipeline;
    @:jsMethod("pipeline-destroy")
    public function destroy():Void;
}

@:jsImport("rayzor-gpu")
extern class CmdEncoder {
    @:jsMethod("cmd-create")
    public static function create():CmdEncoder;
    @:jsMethod("cmd-begin-pass")
    public function beginPass(colorView:Int, loadOp:Int, r:Float, g:Float, b:Float, a:Float, depthView:Int):Void;
    @:jsMethod("cmd-set-pipeline")
    public function setPipeline(pipeline:Pipeline):Void;
    @:jsMethod("cmd-draw")
    public function draw(vertexCount:Int, instanceCount:Int, firstVertex:Int, firstInstance:Int):Void;
    @:jsMethod("cmd-end-pass")
    public function endPass():Void;
    @:jsMethod("cmd-submit")
    public function submit(device:GPUDevice):Void;
    @:jsMethod("cmd-destroy")
    public function destroy():Void;
}
#else
import rayzor.window.Window;
import rayzor.window.Key;
import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;
#end

class Main {
    static function main() {
        trace("=== GPU Window Demo ===");

        #if wasm
        if (!GPUDevice.isAvailable()) {
            trace("GPU not available");
            return;
        }

        var win = Window.createCentered("Rayzor GPU", 800, 600);
        var device = GPUDevice.create();
        var surface = GfxSurface.createCanvas(device, null, 800, 600);
        trace("Surface format: " + surface.getFormat());

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
        var shader = Shader.create(device, wgsl, "vs_main", "fs_main");
        var pipe = Pipeline.begin();
        pipe.setShader(shader);
        pipe.setFormat(surface.getFormat());
        var pipeline = pipe.build(device);
        trace("Pipeline ready");

        var cmd = CmdEncoder.create();
        var frames = 0;

        Window.runLoop(win, function():Bool {
            if (win.wasResized()) {
                surface.resize(device, win.getWidth(), win.getHeight());
            }
            var view = surface.getTexture();
            if (view == 0) return true;

            cmd.beginPass(view, 0, 0.05, 0.05, 0.15, 1.0, 0);
            cmd.setPipeline(pipeline);
            cmd.draw(3, 1, 0, 0);
            cmd.endPass();
            cmd.submit(device);
            surface.present();
            frames++;
            if (frames % 120 == 0) trace("Frame " + frames);
            return true;
        });
        #else
        // Native path — uses rpkg extern classes
        if (!GPUDevice.isAvailable()) {
            trace("GPU not available");
            return;
        }
        var win = Window.createCentered("Rayzor GPU", 800, 600);
        var device = GPUDevice.create();
        var surface = Surface.create(device, win.getHandle(), win.getDisplayHandle(), 800, 600);
        trace("Surface format: " + surface.getFormat());

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
        trace("Pipeline ready");

        var cmd = CommandEncoder.create();
        var frames = 0;
        while (win.pollEvents()) {
            if (win.isKeyDown(Key.ESCAPE)) break;
            if (win.wasResized()) surface.resize(device, win.getWidth(), win.getHeight());
            var view = surface.getTexture();
            if (view == null) continue;
            cmd.beginPass(view, 0, 0.05, 0.05, 0.15, 1.0, null);
            cmd.setPipeline(pipeline);
            cmd.draw(3, 1, 0, 0);
            cmd.endPass();
            cmd.submit(device);
            surface.present();
            frames++;
            if (frames % 120 == 0) trace("Frame " + frames);
        }
        trace("Done — " + frames + " frames");
        pipeline.destroy();
        shader.destroy();
        surface.destroy();
        device.destroy();
        win.destroy();
        #end
    }
}
