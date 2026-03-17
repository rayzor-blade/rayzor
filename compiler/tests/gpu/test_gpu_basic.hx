// GPU Basic Operations E2E Test
// Tests: buffer lifecycle, elementwise ops, reductions, dot, matmul
// Run: rayzor run compiler/tests/gpu/test_gpu_basic.hx --rpkg <rayzor-gpu.rpkg>

import rayzor.gpu.GPUCompute;
import rayzor.gpu.GpuBuffer;
import rayzor.ds.Tensor;

class Main {
    static var passed = 0;
    static var failed = 0;

    static function check(name:String, got:Float, expected:Float):Void {
        if (got == expected) {
            passed = passed + 1;
        } else {
            trace("FAIL: " + name + " — expected " + expected + ", got " + got);
            failed = failed + 1;
        }
    }

    static function checkInt(name:String, got:Int, expected:Int):Void {
        if (got == expected) {
            passed = passed + 1;
        } else {
            trace("FAIL: " + name + " — expected " + expected + ", got " + got);
            failed = failed + 1;
        }
    }

    static function main() {
        var available = GPUCompute.isAvailable();
        if (!available) {
            trace("SKIP: GPU not available");
            return;
        }

        var gpu = GPUCompute.create();

        // --- Buffer lifecycle ---
        var t = Tensor.ones([1024], F32);
        var buf = gpu.createBuffer(t);
        checkInt("numel", buf.numel(), 1024);
        var readback = gpu.toTensor(buf);
        check("readback_sum", readback.sum(), 1024.0);

        // --- Binary elementwise ---
        var a = gpu.createBuffer(Tensor.full([512], 3.0, F32));
        var b = gpu.createBuffer(Tensor.full([512], 7.0, F32));

        check("add", gpu.sum(gpu.add(a, b)), 5120.0);    // (3+7)*512 = 5120
        check("sub", gpu.sum(gpu.sub(b, a)), 2048.0);     // (7-3)*512 = 2048
        check("mul", gpu.sum(gpu.mul(a, b)), 10752.0);    // (3*7)*512 = 10752
        check("div", gpu.sum(gpu.div(b, a)), 1194.666748046875);  // (7/3)*512 ≈ 1194.67

        // --- Unary elementwise ---
        check("neg", gpu.sum(gpu.neg(a)), -1536.0);       // -3*512
        check("abs_neg", gpu.sum(gpu.abs(gpu.neg(a))), 1536.0);
        check("relu_pos", gpu.sum(gpu.relu(a)), 1536.0);  // relu(3) = 3
        check("relu_neg", gpu.sum(gpu.relu(gpu.neg(a))), 0.0); // relu(-3) = 0

        // --- Reductions ---
        check("sum", gpu.sum(a), 1536.0);
        check("mean", gpu.mean(a), 3.0);
        check("max", gpu.max(b), 7.0);
        check("min", gpu.min(a), 3.0);

        // --- Dot product ---
        check("dot", gpu.dot(a, b), 10752.0);  // 3*7*512

        // --- Matmul [2x3] * [3x2] ---
        var mA = gpu.createBuffer(Tensor.fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], F32));
        var mB = gpu.createBuffer(Tensor.fromArray([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], F32));
        var mC = gpu.matmul(mA, mB, 2, 3, 2);
        // C = [[58, 64], [139, 154]]
        check("matmul_sum", gpu.sum(mC), 415.0);

        // --- Cleanup ---
        gpu.freeBuffer(buf);
        gpu.freeBuffer(a);
        gpu.freeBuffer(b);
        gpu.freeBuffer(mA);
        gpu.freeBuffer(mB);
        gpu.destroy();

        trace(passed + " passed, " + failed + " failed");
        if (failed == 0) trace("ALL GPU TESTS PASSED");
    }
}
