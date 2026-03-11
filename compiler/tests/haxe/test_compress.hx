class Main {
    static function main() {
        // Test Bytes.ofString + length
        var input = haxe.io.Bytes.ofString("Hello, World! Hello, World! Hello, World! Hello, World!");
        trace(input.length);  // 55

        // Test Compress.run one-shot compression
        var compressed = haxe.zip.Compress.run(input, 6);
        trace(compressed != null);  // true
        trace(compressed.length > 0);  // true
        trace(compressed.length < input.length);  // true — repetitive data compresses well

        // Test streaming Compress API — constructor + close
        var c = new haxe.zip.Compress(6);
        trace(c != null);  // true
        c.close();

        // Test streaming Uncompress API — default constructor (0 args) + close
        var u = new haxe.zip.Uncompress();
        trace(u != null);  // true
        u.close();

        // Test Uncompress with explicit windowBits
        var u2 = new haxe.zip.Uncompress(15);
        trace(u2 != null);  // true
        u2.close();

        trace("done");
    }
}
