// GPU Structured Buffer E2E Test
// Tests: @:gpuStruct layout, createStructBuffer, readStructFloat/Int, gpuDef/gpuSize
// Run: rayzor run compiler/tests/gpu/test_gpu_struct.hx --rpkg <rayzor-gpu.rpkg> --compute

import rayzor.gpu.GPUCompute;
import rayzor.gpu.GpuBuffer;

@:gpuStruct
class Particle {
    public var x:Float;
    public var y:Float;
    public var z:Float;
    public var mass:Float;

    public function new(px:Float, py:Float, pz:Float, m:Float) {
        x = px; y = py; z = pz; mass = m;
    }
}

@:gpuStruct
class Vertex {
    public var posX:Float;
    public var posY:Float;
    public var colorR:Float;
    public var colorG:Float;
    public var colorB:Float;

    public function new(px:Float, py:Float, r:Float, g:Float, b:Float) {
        posX = px; posY = py; colorR = r; colorG = g; colorB = b;
    }
}

class Main {
    static var passed = 0;
    static var failed = 0;

    static function check(name:String, got:Float, expected:Float):Void {
        var diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff < 0.01) {
            passed = passed + 1;
        } else {
            trace("FAIL: " + name + " — expected " + expected + ", got " + got);
            failed = failed + 1;
        }
    }

    static function main() {
        // --- Compile-time layout checks ---
        trace("Particle.gpuDef = " + Particle.gpuDef());
        trace("Particle.gpuSize = " + Particle.gpuSize());
        trace("Particle.gpuAlignment = " + Particle.gpuAlignment());

        check("particle_size", Particle.gpuSize(), 16);       // 4 floats × 4 bytes
        check("particle_align", Particle.gpuAlignment(), 4);

        trace("Vertex.gpuDef = " + Vertex.gpuDef());
        trace("Vertex.gpuSize = " + Vertex.gpuSize());
        check("vertex_size", Vertex.gpuSize(), 20);            // 5 floats × 4 bytes

        if (!GPUCompute.isAvailable()) {
            trace("SKIP: GPU not available for buffer tests");
            trace(passed + " passed, " + failed + " failed");
            return;
        }

        var gpu = GPUCompute.create();

        // --- Create particles on CPU, upload to GPU, read back ---
        var p0 = new Particle(1.0, 2.0, 3.0, 10.0);
        var p1 = new Particle(4.0, 5.0, 6.0, 20.0);

        // Use allocStructBuffer + manual field readback
        var structSize = Particle.gpuSize();
        var buf = gpu.allocStructBuffer(2, structSize);

        // Read from an uploaded buffer
        // First create from raw data
        var particles = new Array<Dynamic>();
        particles.push(p0);
        particles.push(p1);
        var uploaded = gpu.createStructBuffer(particles, 2, structSize);

        // Read fields back
        check("p0_x", gpu.readStructFloat(uploaded, 0, structSize, 0), 1.0);
        check("p0_y", gpu.readStructFloat(uploaded, 0, structSize, 4), 2.0);
        check("p0_z", gpu.readStructFloat(uploaded, 0, structSize, 8), 3.0);
        check("p0_mass", gpu.readStructFloat(uploaded, 0, structSize, 12), 10.0);
        check("p1_x", gpu.readStructFloat(uploaded, 1, structSize, 0), 4.0);
        check("p1_mass", gpu.readStructFloat(uploaded, 1, structSize, 12), 20.0);

        gpu.freeBuffer(buf);
        gpu.freeBuffer(uploaded);
        gpu.destroy();

        trace(passed + " passed, " + failed + " failed");
        if (failed == 0) trace("ALL GPU STRUCT TESTS PASSED");
    }
}
