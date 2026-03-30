/**
 * Tensor demo — vector dot product and array math.
 *
 * Run: rayzor build --target wasm --browser examples/wasm-features/TensorDemo.hx
 */
class TensorDemo {
    static function main() {
        trace("=== Tensor Demo ===");

        // Array as simple vector
        var v = [1, 2, 3, 4];
        trace("Vector: length=" + v.length);

        // Manual dot product (integer)
        var w = [5, 6, 7, 8];
        var dot = 0;
        var i = 0;
        while (i < v.length) {
            dot += v[i] * w[i];
            i++;
        }
        trace("dot([1,2,3,4], [5,6,7,8]) = " + dot); // 70

        // Sum
        var sum = 0;
        i = 0;
        while (i < v.length) {
            sum += v[i];
            i++;
        }
        trace("sum([1,2,3,4]) = " + sum); // 10

        trace("=== Done ===");
    }
}
