class GenericDynamicStringification {
    static function render<T>(value:T):String {
        return Std.string(value);
    }

    static function main() {
        var word:Dynamic = "foo";
        var number:Dynamic = 42;
        if (render(word) != "foo") throw "generic Dynamic string";
        if (render(number) != "42") throw "generic Dynamic integer";

        var buf = new StringBuf();
        buf.add(word);
        buf.add(number);
        buf.add("bar");
        if (buf.toString() != "foo42bar") throw "StringBuf generic add";
        Sys.println("CONFORMANCE_OK");
    }
}
