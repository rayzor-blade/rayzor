// GPU Kernel Fusion E2E Test
// Tests: lazy evaluation, fused kernels, chained operations
// Run: rayzor run compiler/tests/gpu/test_gpu_fusion.hx --rpkg <rayzor-gpu.rpkg>

import rayzor.gpu.GPUCompute;
import rayzor.gpu.GpuBuffer;
import rayzor.ds.Tensor;

class Main {
    static var passed = 0;
    static var failed = 0;

    static function check(name:String, got:Float, expected:Float):Void {
        // Allow small floating point error
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
        if (!GPUCompute.isAvailable()) {
            trace("SKIP: GPU not available");
            return;
        }

        var gpu = GPUCompute.create();

        // --- Chained ops (lazy fusion) ---
        // result = (a + b) * c
        var a = gpu.createBuffer(Tensor.full([1024], 2.0, F32));
        var b = gpu.createBuffer(Tensor.full([1024], 3.0, F32));
        var c = gpu.createBuffer(Tensor.full([1024], 4.0, F32));

        var result = gpu.mul(gpu.add(a, b), c);
        check("chain_add_mul", gpu.sum(result), 20480.0);  // (2+3)*4*1024 = 20480

        // --- ReLU chain: relu(a - b) where a < b ---
        var small = gpu.createBuffer(Tensor.full([256], 1.0, F32));
        var big = gpu.createBuffer(Tensor.full([256], 5.0, F32));
        var reluResult = gpu.relu(gpu.sub(small, big));
        check("relu_chain", gpu.sum(reluResult), 0.0);  // relu(-4) = 0

        // --- Absolute of negation chain ---
        var vals = gpu.createBuffer(Tensor.full([128], 7.0, F32));
        var absNeg = gpu.abs(gpu.neg(vals));
        check("abs_neg_chain", gpu.sum(absNeg), 896.0);  // |(-7)| * 128 = 896

        // --- Exp + Log roundtrip (should be identity) ---
        var ones = gpu.createBuffer(Tensor.full([64], 1.0, F32));
        var roundtrip = gpu.log(gpu.exp(ones));
        check("exp_log_roundtrip", gpu.sum(roundtrip), 64.0);  // log(e^1) = 1 → sum = 64

        // --- Sqrt ---
        var nines = gpu.createBuffer(Tensor.full([100], 9.0, F32));
        check("sqrt", gpu.sum(gpu.sqrt(nines)), 300.0);  // sqrt(9)*100 = 300

        // --- Div chain ---
        var tens = gpu.createBuffer(Tensor.full([200], 10.0, F32));
        var twos = gpu.createBuffer(Tensor.full([200], 2.0, F32));
        check("div_chain", gpu.sum(gpu.div(tens, twos)), 1000.0);  // (10/2)*200 = 1000

        // Cleanup
        gpu.freeBuffer(a);
        gpu.freeBuffer(b);
        gpu.freeBuffer(c);
        gpu.destroy();

        trace(passed + " passed, " + failed + " failed");
        if (failed == 0) trace("ALL FUSION TESTS PASSED");
    }
}
