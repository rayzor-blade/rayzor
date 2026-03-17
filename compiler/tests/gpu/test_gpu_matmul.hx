// GPU Matmul E2E Test
// Tests: matrix multiplication correctness with various sizes
// Run: rayzor run compiler/tests/gpu/test_gpu_matmul.hx --rpkg <rayzor-gpu.rpkg>

import rayzor.gpu.GPUCompute;
import rayzor.gpu.GpuBuffer;
import rayzor.ds.Tensor;

class Main {
    static var passed = 0;
    static var failed = 0;

    static function check(name:String, got:Float, expected:Float):Void {
        var diff = got - expected;
        if (diff < 0) diff = -diff;
        if (diff < 0.5) {
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

        // --- Identity matmul: A * I = A ---
        // A = [[1,2],[3,4]], I = [[1,0],[0,1]]
        var a = gpu.createBuffer(Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32));
        var eye = gpu.createBuffer(Tensor.fromArray([1.0, 0.0, 0.0, 1.0], F32));
        var ai = gpu.matmul(a, eye, 2, 2, 2);
        check("identity_sum", gpu.sum(ai), 10.0);  // 1+2+3+4 = 10

        // --- Square matmul: [2x2] * [2x2] ---
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //   = [[19, 22], [43, 50]]
        var b = gpu.createBuffer(Tensor.fromArray([5.0, 6.0, 7.0, 8.0], F32));
        var c = gpu.matmul(a, b, 2, 2, 2);
        check("2x2_sum", gpu.sum(c), 134.0);  // 19+22+43+50 = 134

        // --- Rectangular: [1x3] * [3x1] = [1x1] (inner product) ---
        var row = gpu.createBuffer(Tensor.fromArray([1.0, 2.0, 3.0], F32));
        var col = gpu.createBuffer(Tensor.fromArray([4.0, 5.0, 6.0], F32));
        var inner = gpu.matmul(row, col, 1, 3, 1);
        check("inner_product", gpu.sum(inner), 32.0);  // 1*4+2*5+3*6 = 32

        // --- Rectangular: [3x1] * [1x3] = [3x3] (outer product) ---
        var outer = gpu.matmul(col, row, 3, 1, 3);
        // [[4,8,12],[5,10,15],[6,12,18]]
        check("outer_product", gpu.sum(outer), 90.0);  // sum = 4+8+12+5+10+15+6+12+18 = 90

        // --- Larger: [4x4] * [4x4] ---
        var big = gpu.createBuffer(Tensor.ones([16], F32));
        var bigR = gpu.matmul(big, big, 4, 4, 4);
        check("4x4_ones", gpu.sum(bigR), 64.0);  // each element = 4, 16 elements → 64

        // Cleanup
        gpu.freeBuffer(a);
        gpu.freeBuffer(b);
        gpu.destroy();

        trace(passed + " passed, " + failed + " failed");
        if (failed == 0) trace("ALL MATMUL TESTS PASSED");
    }
}
