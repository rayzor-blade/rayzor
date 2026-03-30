import rayzor.Tensor;

/**
 * Tensor demo — construction, elementwise, matmul, reductions.
 * CPU TypedArray engine with WASM SIMD acceleration.
 * GPU offload via WebGPU when rayzor-gpu host is available.
 *
 * Run: rayzor build --target wasm --browser examples/wasm-features/TensorDemo.hx
 */
class TensorDemo {
    static function main() {
        trace("=== Tensor Demo ===");

        // Create tensors
        var a = Tensor.ones([2, 3]);
        trace("ones(2x3): numel=" + a.numel());

        var b = Tensor.full([2, 3], 2.0);
        trace("full(2x3, 2.0): element=" + b.get(0));

        // Elementwise
        var c = a.add(b);
        trace("ones + full(2) = " + c.get(0));  // 3.0

        // Reductions
        trace("sum(ones 2x3) = " + a.sum());    // 6.0
        trace("mean(full 2x3) = " + b.mean());  // 2.0

        // Dot product
        var v1 = Tensor.fromArray([1.0, 2.0, 3.0]);
        var v2 = Tensor.fromArray([4.0, 5.0, 6.0]);
        trace("dot([1,2,3], [4,5,6]) = " + v1.dot(v2));  // 32.0

        // Matrix multiply
        var m1 = Tensor.fromArray([1.0, 2.0, 3.0, 4.0]).reshape([2, 2]);
        var m2 = Tensor.fromArray([5.0, 6.0, 7.0, 8.0]).reshape([2, 2]);
        var m3 = m1.matmul(m2);
        trace("matmul = [" + m3.get(0) + "," + m3.get(1) + "," + m3.get(2) + "," + m3.get(3) + "]");

        // Unary
        var vals = Tensor.fromArray([1.0, 4.0, 9.0, 16.0]);
        var sq = vals.sqrt();
        trace("sqrt = [" + sq.get(0) + "," + sq.get(1) + "," + sq.get(2) + "," + sq.get(3) + "]");

        trace("=== Done ===");
    }
}
