enum Color {
    Red;
    Green;
    Blue;
}

class Main {
    static function main() {
        // Test createEnumIndex - simple unboxed
        var red = Type.createEnumIndex(Color, 0, null);
        trace(red);   // 0

        var green = Type.createEnum(Color, "Green", null);
        trace(green); // 1

        var blue = Type.createEnumIndex(Color, 2, null);
        trace(blue);  // 2
    }
}
