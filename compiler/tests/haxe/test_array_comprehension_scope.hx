class Main {
    static function main() {
        var a = [for (i in 0...3) i];
        var b = [for (i in 0...3) i + 10];

        trace(a[0]); // 0
        trace(a[1]); // 1
        trace(a[2]); // 2

        trace(b[0]); // 10
        trace(b[1]); // 11
        trace(b[2]); // 12

        var src = [1, 2, 3];
        var c = [for (x in src) x * 2];
        trace(c[0]); // 2
        trace(c[1]); // 4
        trace(c[2]); // 6
    }
}
