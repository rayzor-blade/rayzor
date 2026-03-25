import rayzor.runtime.CC;
import rayzor.Ptr;
import rayzor.gpu.GPUDevice;
import rayzor.gpu.Surface;
import rayzor.gpu.ShaderModule;
import rayzor.gpu.RenderPipeline;
import rayzor.gpu.CommandEncoder;

/**
 * GPU Window Rendering — renders a triangle to a native window.
 *
 * Uses TinyCC for window creation (Cocoa on macOS) and rayzor GPU
 * for rendering. Everything written in Haxe.
 *
 * Run: rayzor run --rpkg rayzor-gpu.rpkg examples/gpu-window/Main.hx
 */
class Main {
    static function main() {
        // 1. Create native window + event loop via TinyCC + Cocoa
        var cc = CC.create();
        cc.addFramework("Cocoa");
        cc.compile('
            #include <objc/runtime.h>
            #include <objc/message.h>
            typedef unsigned long NSUInteger;
            typedef double CGFloat;
            typedef struct { CGFloat x, y, w, h; } CGRect;
            static id g_window = 0;

            long create_window(void) {
                id app = ((id(*)(id, SEL))objc_msgSend)((id)objc_getClass("NSApplication"), sel_registerName("sharedApplication"));
                ((void(*)(id, SEL, NSUInteger))objc_msgSend)(app, sel_registerName("setActivationPolicy:"), 0);
                CGRect frame = {100.0, 100.0, 800.0, 600.0};
                id alloc = ((id(*)(id, SEL))objc_msgSend)((id)objc_getClass("NSWindow"), sel_registerName("alloc"));
                g_window = ((id(*)(id, SEL, CGRect, NSUInteger, NSUInteger, BOOL))objc_msgSend)(alloc, sel_registerName("initWithContentRect:styleMask:backing:defer:"), frame, (NSUInteger)15, (NSUInteger)2, 0);
                id title = ((id(*)(id, SEL, const char*))objc_msgSend)((id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), "Rayzor");
                ((void(*)(id, SEL, id))objc_msgSend)(g_window, sel_registerName("setTitle:"), title);
                ((void(*)(id, SEL, id))objc_msgSend)(g_window, sel_registerName("makeKeyAndOrderFront:"), (id)0);
                ((void(*)(id, SEL, BOOL))objc_msgSend)(app, sel_registerName("activateIgnoringOtherApps:"), 1);
                id view = ((id(*)(id, SEL))objc_msgSend)(g_window, sel_registerName("contentView"));
                ((void(*)(id, SEL, BOOL))objc_msgSend)(view, sel_registerName("setWantsLayer:"), 1);
                return (long)view;
            }

            long poll_events(void) {
                id app = ((id(*)(id, SEL))objc_msgSend)((id)objc_getClass("NSApplication"), sel_registerName("sharedApplication"));
                while (1) {
                    id event = ((id(*)(id, SEL, NSUInteger, id, id, BOOL))objc_msgSend)(
                        app, sel_registerName("nextEventMatchingMask:untilDate:inMode:dequeue:"),
                        (NSUInteger)~(NSUInteger)0, (id)0,
                        ((id(*)(id, SEL, const char*))objc_msgSend)((id)objc_getClass("NSString"), sel_registerName("stringWithUTF8String:"), "kCFRunLoopDefaultMode"),
                        1);
                    if (!event) break;
                    ((void(*)(id, SEL, id))objc_msgSend)(app, sel_registerName("sendEvent:"), event);
                }
                return (long)((id(*)(id, SEL))objc_msgSend)(g_window, sel_registerName("isVisible"));
            }
        ');
        cc.relocate();
        var viewPtr = CC.call0(cc.getSymbol("create_window"));
        var pollFn = cc.getSymbol("poll_events");
        trace("Window created");

        // 2. GPU device + surface
        var device = GPUDevice.create();
        var surface = Surface.create(device, viewPtr, untyped cast 0, 800, 600);
        trace("Surface ready");

        // 3. Shader + pipeline
        var wgsl = "
            @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
                var pos = array<vec2f, 3>(
                    vec2f( 0.0,  0.5),
                    vec2f(-0.5, -0.5),
                    vec2f( 0.5, -0.5));
                return vec4f(pos[vi], 0.0, 1.0);
            }
            @fragment fn fs() -> @location(0) vec4f {
                return vec4f(1.0, 0.4, 0.1, 1.0);
            }";
        var shader = ShaderModule.create(device, wgsl, "vs", "fs");
        var pipe = RenderPipeline.begin();
        pipe.setShader(shader);
        pipe.setFormat(surface.getFormat());
        var built = pipe.build(device);
        trace("Rendering...");

        // 4. Render loop — present frames until window closes or 300 frames
        var frames = 0;
        var running = true;
        while (running && frames < 300) {
            // Poll window events — check if window is still visible
            if (CC.call0(pollFn) == null) {
                running = false;
                break;
            }

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

        // Cleanup
        built.destroy();
        shader.destroy();
        surface.destroy();
        device.destroy();
    }
}
