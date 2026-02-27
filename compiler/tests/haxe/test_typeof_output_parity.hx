enum Color {
    Red;
}

class Main {
    static function main() {
        var vtFn = Type.typeof(main);
        var vtStr = Type.typeof("x");
        var vtEnum = Type.typeof(Color.Red);

        // Direct trace path
        trace(Type.typeof(main));
        trace(Type.typeof("x"));
        trace(Type.typeof(Color.Red));

        // Stored variable path
        trace(vtFn);
        trace(vtStr);
        trace(vtEnum);

        // Std.string path
        trace(Std.string(vtFn));
        trace(Std.string(vtStr));
        trace(Std.string(vtEnum));

        // Interpolation path
        trace('${vtFn}');
        trace('${vtStr}');
        trace('${vtEnum}');
    }
}
