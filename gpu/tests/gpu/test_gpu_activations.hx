// GPU Activation Functions E2E Test
// Tests: sigmoid, tanh, gelu, silu + fusion chains
// Run: rayzor run compiler/tests/gpu/test_gpu_activations.hx --rpkg <rayzor-gpu.rpkg>

import rayzor.gpu.GPUCompute;
import rayzor.gpu.GpuBuffer;
import rayzor.ds.Tensor;

class Main {
    static var passed = 0;
    static var failed = 0;

    static function checkRange(name:String, got:Float, lo:Float, hi:Float):Void {
        if (got >= lo && got <= hi) {
            passed = passed + 1;
        } else {
            trace("FAIL: " + name + " — got " + got + " not in [" + lo + "," + hi + "]");
            failed = failed + 1;
        }
    }

    static function check(name:String, got:Float, expected:Float):Void {
        var diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff < 0.1) {
            passed = passed + 1;
        } else {
            trace("FAIL: " + name + " — expected " + expected + ", got " + got);
            failed = failed + 1;
        }
    }

    static function main() {
        if (!GPUCompute.isAvailable()) { trace("SKIP: GPU not available"); return; }

        var gpu = GPUCompute.create();
        var n = 256;

        // --- Sigmoid: 1/(1+exp(-x)) ---
        // sigmoid(0) = 0.5, so sum of 256 zeros through sigmoid = 128
        var zeros = gpu.createBuffer(Tensor.full([n], 0.0, F32));
        check("sigmoid_zero", gpu.sum(gpu.sigmoid(zeros)), 128.0);

        // sigmoid(large) ≈ 1, sigmoid(-large) ≈ 0
        var pos = gpu.createBuffer(Tensor.full([n], 10.0, F32));
        var neg = gpu.createBuffer(Tensor.full([n], -10.0, F32));
        checkRange("sigmoid_pos", gpu.mean(gpu.sigmoid(pos)), 0.99, 1.01);
        checkRange("sigmoid_neg", gpu.mean(gpu.sigmoid(neg)), -0.01, 0.01);

        // --- Tanh ---
        // tanh(0) = 0
        check("tanh_zero", gpu.sum(gpu.tanh(zeros)), 0.0);
        // tanh(large) ≈ 1
        checkRange("tanh_pos", gpu.mean(gpu.tanh(pos)), 0.99, 1.01);
        // tanh(-large) ≈ -1
        checkRange("tanh_neg", gpu.mean(gpu.tanh(neg)), -1.01, -0.99);

        // --- GELU ---
        // gelu(0) = 0
        check("gelu_zero", gpu.sum(gpu.gelu(zeros)), 0.0);
        // gelu(large) ≈ x (identity for large positive)
        checkRange("gelu_pos_mean", gpu.mean(gpu.gelu(pos)), 9.9, 10.1);
        // gelu(-large) ≈ 0
        checkRange("gelu_neg", gpu.mean(gpu.gelu(neg)), -0.1, 0.1);

        // --- SiLU (Swish): x * sigmoid(x) ---
        // silu(0) = 0 * 0.5 = 0
        check("silu_zero", gpu.sum(gpu.silu(zeros)), 0.0);
        // silu(large) ≈ x
        checkRange("silu_pos_mean", gpu.mean(gpu.silu(pos)), 9.9, 10.1);

        // --- Fusion: sigmoid(relu(x)) ---
        var mixed = gpu.createBuffer(Tensor.fromArray([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], F32));
        var fused = gpu.sigmoid(gpu.relu(mixed));
        // relu: [0, 0, 0, 1, 2, 3] → sigmoid: [0.5, 0.5, 0.5, 0.73, 0.88, 0.95]
        checkRange("fusion_sigmoid_relu", gpu.sum(fused), 3.5, 4.5);

        gpu.freeBuffer(zeros);
        gpu.freeBuffer(pos);
        gpu.freeBuffer(neg);
        gpu.destroy();

        trace(passed + " passed, " + failed + " failed");
        if (failed == 0) trace("ALL ACTIVATION TESTS PASSED");
    }
}
