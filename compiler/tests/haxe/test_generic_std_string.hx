// Std.string / string-concat on a TYPE-ERASED generic parameter.
// A generic call bitcasts its argument to i64, so the callee's type-tag
// fixup must recover the concrete type from the bitcast source. Without
// that, Float rendered its raw f64 bits; before the Std.string routing
// fix, String rendered "<unknown type N>" and Int SIGSEGV'd.
class TestGenericStdString {
    static function viaStdString<T>(x:T):String {
        return Std.string(x);
    }

    static function viaConcat<T>(x:T):String {
        return "" + x;
    }

    public static function main() {
        trace(viaStdString("Hello")); // Hello
        trace(viaStdString(42)); // 42
        trace(viaStdString(1.5)); // 1.5
        trace(viaStdString(true)); // true

        trace(viaConcat("Hello")); // Hello
        trace(viaConcat(42)); // 42
        trace(viaConcat(1.5)); // 1.5
        trace(viaConcat(true)); // true

        // Std.string on a concrete String is the identity, not a Dynamic unbox.
        var s = "World";
        trace(Std.string("Hello")); // Hello
        trace(Std.string(s)); // World

        // StringBuf.add is Std.string(x:T) internally.
        var buf = new StringBuf();
        buf.add("n=");
        buf.add(42);
        buf.add(" f=");
        buf.add(1.5);
        buf.add(" b=");
        buf.add(true);
        trace(buf.toString()); // n=42 f=1.5 b=true
    }
}
