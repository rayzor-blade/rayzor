class UnicodeEscapes {
    static function main() {
        if (haxe.io.Bytes.ofString("\u{1F602}").toHex() != "f09f9882") throw "braced Unicode";
        if (haxe.io.Bytes.ofString("\u{10FFFF}").toHex() != "f48fbfbf") throw "maximum scalar";
        if ("\u{41}\u0042\x43" != "ABC") throw "mixed escapes";
        if ("\x70" != "p") throw "hex escape";
        if ("\\u{1F602}" != '\\u{1F602}') throw "literal escape";
        var suffix = "!";
        var interpolated = 'emoji \u{1F602} $suffix';
        if (haxe.io.Bytes.ofString(interpolated).toHex() != "656d6f6a6920f09f98822021") throw "interpolated escape";
        trace("CONFORMANCE_OK");
    }
}
