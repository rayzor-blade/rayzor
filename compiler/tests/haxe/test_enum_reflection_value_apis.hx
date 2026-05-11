// Behavioural test: Type.enumConstructor / enumIndex / enumParameters
// on tag-only AND mixed enums (the latter has variants with params,
// so all variants compile to heap-boxed `[tag:i32, fields...]`).

enum Maybe {
    None;
    Some(v: Int);
}

enum Color {
    Red;
    Green;
    Blue;
}

class Main {
    static function main() {
        // Mixed enum: Maybe has Some(Int) → heap-boxed representation.
        var s = Some(42);
        trace(Type.enumConstructor(s));    // Some
        trace(Type.enumIndex(s));           // 1
        trace(Type.enumParameters(s));     // [42]

        var n = None;
        trace(Type.enumConstructor(n));    // None
        trace(Type.enumIndex(n));           // 0
        trace(Type.enumParameters(n));     // []

        // Tag-only enum: Color → raw i64 discriminant representation.
        var r = Red;
        trace(Type.enumConstructor(r));    // Red
        trace(Type.enumIndex(r));           // 0

        var b = Blue;
        trace(Type.enumConstructor(b));    // Blue
        trace(Type.enumIndex(b));           // 2
    }
}
