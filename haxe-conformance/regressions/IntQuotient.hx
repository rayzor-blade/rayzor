// `Std.int(a / c)` truncates the Float quotient toward zero; computed in
// integers for a constant `c`, it must give the same Int as the Float path.
class Main {
    static inline var BASE = 10000;

    // Read at run time, so the reference keeps the Float division.
    static var divisors = [BASE, 7, -3, 1, 2147483647, -2147483648, 65536];

    static function viaFloat(a:Int, i:Int):Int {
        return Std.int(a / divisors[i]);
    }

    static function check(a:Int) {
        var cases = [
            [Std.int(a / BASE), viaFloat(a, 0)],
            [Std.int(a / 7), viaFloat(a, 1)],
            [Std.int(a / -3), viaFloat(a, 2)],
            [Std.int(a / 1), viaFloat(a, 3)],
            [Std.int(a / 2147483647), viaFloat(a, 4)],
            [Std.int(a / -2147483648), viaFloat(a, 5)],
            [Std.int(a / 65536), viaFloat(a, 6)],
        ];
        for (i in 0...cases.length)
            if (cases[i][0] != cases[i][1]) throw 'Std.int($a / c) case $i: ${cases[i][0]} != ${cases[i][1]}';
    }

    static function main() {
        var values = [0, 1, -1, 6, -6, 7, -7, 9999, 10000, 10001, -9999, -10000, -10001, 123456789,
            -123456789, 2147483647, -2147483647, -2147483648, 65535, 65536, -65536];
        for (v in values)
            check(Std.parseInt(Std.string(v)));
        var carry = 0;
        for (i in 0...1000) {
            var v = i * 98765 + carry;
            carry = Std.int(v / BASE);
            if (carry != viaFloat(v, 0)) throw "carry " + i;
        }
        Sys.println("CONFORMANCE_OK");
    }
}
