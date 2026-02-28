import rayzor.SIMD4f;

class Main {
    static function main() {
        var a:SIMD4f = [1.0, 2.0, 3.0, 4.0];
        var s = a.sum();
        trace(s);
    }
}
