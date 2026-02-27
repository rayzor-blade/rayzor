enum Color {
    Red;
    Green;
    Blue;
}

enum Maybe {
    None;
    Some(v:Int);
}

class Main {
    static function main() {
        var enumType = Type.getEnum(Color.Green);
        trace(Type.getEnumName(enumType)); // Color

        var resolved = Type.resolveEnum("Color");
        trace(Type.getEnumName(resolved)); // Color

        var all:Array<Dynamic> = Type.allEnums(Color);
        trace(all.length); // 3

        trace(Type.enumEq(Color.Red, Color.Red)); // true
        trace(Type.enumEq(Color.Red, Color.Blue)); // false

        var s1 = Maybe.Some(7);
        var s2 = Maybe.Some(7);
        var s3 = Maybe.Some(8);
        var n = Maybe.None;

        trace(Type.enumEq(s1, s2)); // true
        trace(Type.enumEq(s1, s3)); // false
        trace(Type.enumEq(s1, n));  // false
    }
}
