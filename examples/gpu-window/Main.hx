import rayzor.runtime.CC;
import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;
import rayzor.gpu.Vec2;
import rayzor.gpu.Vec3;
import rayzor.gpu.Vec4;

/**
 * GPU Window Rendering Example
 *
 * Uses TinyCC to request a window from the OS, then renders a colored
 * triangle to the window surface using the rayzor GPU API.
 * Shader, window, rendering — all written in Haxe.
 *
 * Run:
 *   rayzor run --rpkg rayzor-gpu.rpkg examples/gpu-window/Main.hx
 */

@:shader
class TriangleShader {
    @:vertex
    function vertex(vertexIndex:Int):VOut {
        var positions = [
            Vec2( 0.0,  0.5),
            Vec2(-0.5, -0.5),
            Vec2( 0.5, -0.5),
        ];
        var colors = [
            Vec3(1.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
            Vec3(0.0, 0.0, 1.0),
        ];
        var out = new VOut();
        out.position = Vec4(positions[vertexIndex], 0.0, 1.0);
        out.color = colors[vertexIndex];
        return out;
    }

    @:fragment
    function fragment(input:VOut):Vec4 {
        return Vec4(input.color, 1.0);
    }
}

@:gpuStruct
class VOut {
    @:builtin("position") public var position:Vec4;
    public var color:Vec3;
}

class Main {
    static var pollFn:Int = 0;
    static var destroyFn:Int = 0;

    static function createCocoaWindow(width:Int, height:Int):Int {
        var cc = CC.create();
        cc.addFramework("Cocoa");
        cc.compile('
            #include <objc/runtime.h>
            #include <objc/message.h>

            typedef unsigned long NSUInteger;
            typedef double CGFloat;
            typedef struct { CGFloat x, y, w, h; } CGRect;

            static id g_window = 0;
            static id g_view = 0;

            extern long __arg0;
            extern long __arg1;

            long create_window(void) {
                long w = __arg0;
                long h = __arg1;

                id app = ((id(*)(id, SEL))objc_msgSend)(
                    (id)objc_getClass("NSApplication"),
                    sel_registerName("sharedApplication"));
                ((void(*)(id, SEL, NSUInteger))objc_msgSend)(
                    app, sel_registerName("setActivationPolicy:"), 0);

                CGRect frame = {100.0, 100.0, (CGFloat)w, (CGFloat)h};
                id alloc = ((id(*)(id, SEL))objc_msgSend)(
                    (id)objc_getClass("NSWindow"), sel_registerName("alloc"));
                g_window = ((id(*)(id, SEL, CGRect, NSUInteger, NSUInteger, BOOL))objc_msgSend)(
                    alloc,
                    sel_registerName("initWithContentRect:styleMask:backing:defer:"),
                    frame, (NSUInteger)15, (NSUInteger)2, 0);

                id title = ((id(*)(id, SEL, const char*))objc_msgSend)(
                    (id)objc_getClass("NSString"),
                    sel_registerName("stringWithUTF8String:"), "Rayzor");
                ((void(*)(id, SEL, id))objc_msgSend)(
                    g_window, sel_registerName("setTitle:"), title);
                ((void(*)(id, SEL, id))objc_msgSend)(
                    g_window, sel_registerName("makeKeyAndOrderFront:"), (id)0);
                ((void(*)(id, SEL, BOOL))objc_msgSend)(
                    app, sel_registerName("activateIgnoringOtherApps:"), 1);

                g_view = ((id(*)(id, SEL))objc_msgSend)(
                    g_window, sel_registerName("contentView"));
                ((void(*)(id, SEL, BOOL))objc_msgSend)(
                    g_view, sel_registerName("setWantsLayer:"), 1);

                return (long)g_view;
            }

            long poll_events(void) {
                id app = ((id(*)(id, SEL))objc_msgSend)(
                    (id)objc_getClass("NSApplication"),
                    sel_registerName("sharedApplication"));
                while (1) {
                    id event = ((id(*)(id, SEL, NSUInteger, id, id, BOOL))objc_msgSend)(
                        app, sel_registerName("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                        (NSUInteger)~(NSUInteger)0, (id)0,
                        ((id(*)(id, SEL, const char*))objc_msgSend)(
                            (id)objc_getClass("NSString"),
                            sel_registerName("stringWithUTF8String:"),
                            "kCFRunLoopDefaultMode"),
                        1);
                    if (!event) break;
                    ((void(*)(id, SEL, id))objc_msgSend)(
                        app, sel_registerName("sendEvent:"), event);
                }
                return (long)((id(*)(id, SEL))objc_msgSend)(
                    g_window, sel_registerName("isVisible"));
            }

            long destroy_window(void) {
                if (g_window) {
                    ((void(*)(id, SEL))objc_msgSend)(
                        g_window, sel_registerName("close"));
                    g_window = 0;
                    g_view = 0;
                }
                return 0;
            }
        ');

        cc.addSymbol("__arg0", width);
        cc.addSymbol("__arg1", height);
        cc.relocate();

        var viewPtr = CC.call0(cc.getSymbol("create_window"));
        pollFn = cc.getSymbol("poll_events");
        destroyFn = cc.getSymbol("destroy_window");

        // Don't delete — JIT code must stay alive for poll/destroy calls
        return viewPtr;
    }

    static function main() {
        var width = 800;
        var height = 600;

        // 1. Request window from OS via TinyCC + Cocoa
        trace("Creating window...");
        var viewPtr = createCocoaWindow(width, height);
        trace("Window created (NSView=" + viewPtr + ")");

        // 2. GPU setup
        var device = GPUDevice.create();
        var surface = Surface.create(device, viewPtr, 0, width, height);
        trace("GPU surface ready");

        // 3. Shader + pipeline
        var shader = ShaderModule.create(device, TriangleShader.wgsl(), "vertex", "fragment");
        var pipe = RenderPipeline.begin();
        pipe.setShader(shader);
        pipe.setFormat(surface.getFormat());
        var built = pipe.build(device);
        trace("Rendering...");

        // 4. Render loop
        var frames = 0;
        while (CC.call0(pollFn) != 0) {
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

            if (frames >= 600) break;
        }

        trace("Rendered " + frames + " frames");
        built.destroy();
        shader.destroy();
        surface.destroy();
        device.destroy();
        CC.call0(destroyFn);
    }
}
