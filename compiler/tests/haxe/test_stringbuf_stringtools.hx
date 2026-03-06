// Test StringBuf and StringTools
// Tests: StringBuf (append, length, toString), StringTools (trim, replace, startsWith, endsWith, contains)

class TestStringBufStringTools {
    public static function main() {
        // === StringBuf basics ===
        var buf = new StringBuf();
        buf.add("Hello");
        buf.add(" ");
        buf.add("World");
        trace(buf.toString()); // Hello World
        trace(buf.length); // 11

        buf.addChar(33); // '!'
        trace(buf.toString()); // Hello World!

        // === StringTools via static calls ===
        // trim
        var padded = "  hello  ";
        trace(StringTools.trim(padded)); // hello
        trace(StringTools.ltrim(padded)); // hello
        trace(StringTools.rtrim(padded)); // ..hello

        // contains
        trace(StringTools.contains("hello world", "world")); // true
        trace(StringTools.contains("hello world", "xyz")); // false

        // startsWith / endsWith
        trace(StringTools.startsWith("hello world", "hello")); // true
        trace(StringTools.startsWith("hello world", "world")); // false
        trace(StringTools.endsWith("hello world", "world")); // true
        trace(StringTools.endsWith("hello world", "hello")); // false

        // replace
        trace(StringTools.replace("hello world", "world", "haxe")); // hello haxe
        trace(StringTools.replace("aaa", "a", "bb")); // bbbbbb

        // isSpace
        trace(StringTools.isSpace(" ", 0)); // true
        trace(StringTools.isSpace("a", 0)); // false

        // hex
        trace(StringTools.hex(255, 4)); // 00FF
        trace(StringTools.hex(16)); // 10

        // lpad / rpad
        trace(StringTools.lpad("42", "0", 5)); // 00042
        trace(StringTools.rpad("hi", ".", 5)); // hi...
    }
}
