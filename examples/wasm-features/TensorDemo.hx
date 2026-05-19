import rayzor.ds.Tensor;
import rayzor.ds.DType;

/**
 * Tensor demo — construction, elementwise, matmul, reductions.
 * CPU TypedArray engine with WASM SIMD acceleration.
 * GPU offload via WebGPU when rayzor-gpu host is available.
 *
 * Run: rayzor run --wasm TensorDemo.hx
 * Build: rayzor build --target wasm --browser TensorDemo.hx
 */
class TensorDemo {
    static function main() {
        trace("=== Tensor Demo ===");

        // Create tensors
        var a = Tensor.ones([2, 3], F32);
        trace("ones(2x3): numel=" + a.numel());

        var b = Tensor.full([2, 3], 2.0, F32);
        trace("full(2x3, 2.0): element=" + b.get([0, 0]));

        // Elementwise
        var c = a.add(b);
        trace("ones + full(2) = " + c.get([0, 0]));  // 3.0

        // Reductions
        trace("sum(ones 2x3) = " + a.sum());    // 6.0
        trace("mean(full 2x3) = " + b.mean());  // 2.0

        // Dot product
        var v1 = Tensor.fromArray([1.0, 2.0, 3.0], F32);
        var v2 = Tensor.fromArray([4.0, 5.0, 6.0], F32);
        trace("dot([1,2,3], [4,5,6]) = " + v1.dot(v2));  // 32.0

        // Matrix multiply
        var m1 = Tensor.fromArray([1.0, 2.0, 3.0, 4.0], F32).reshape([2, 2]);
        var m2 = Tensor.fromArray([5.0, 6.0, 7.0, 8.0], F32).reshape([2, 2]);
        var m3 = m1.matmul(m2);
        trace("matmul = [" + m3.get([0, 0]) + "," + m3.get([0, 1])
              + "," + m3.get([1, 0]) + "," + m3.get([1, 1]) + "]");

        // Unary
        var vals = Tensor.fromArray([1.0, 4.0, 9.0, 16.0], F32);
        var sq = vals.sqrt();
        trace("sqrt = [" + sq.get([0]) + "," + sq.get([1])
              + "," + sq.get([2]) + "," + sq.get([3]) + "]");

        trace("=== Done ===");
    }
}
