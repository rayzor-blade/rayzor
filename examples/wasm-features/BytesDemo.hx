import haxe.io.Bytes;

/**
 * Bytes demo — binary data operations via DataView on WASM linear memory.
 *
 * Run: rayzor run --wasm examples/wasm-features/BytesDemo.hx
 */
class BytesDemo {
    static function main() {
        trace("=== Bytes Demo ===");

        // Allocate and fill
        var buf = Bytes.alloc(16);
        trace("Allocated " + buf.length + " bytes");

        // Write typed values
        buf.setInt32(0, 0x12345678);
        buf.setFloat(4, 3.14);
        buf.setDouble(8, 2.71828);

        // Read back
        trace("Int32 at 0: " + StringTools.hex(buf.getInt32(0)));
        trace("Float at 4: " + buf.getFloat(4));
        trace("Double at 8: " + buf.getDouble(8));

        // Byte access
        buf.set(0, 65); // 'A'
        buf.set(1, 66); // 'B'
        buf.set(2, 67); // 'C'
        trace("Bytes [0..3]: " + buf.get(0) + ", " + buf.get(1) + ", " + buf.get(2));

        // From string
        var str = Bytes.ofString("Hello WASM!");
        trace("String bytes length: " + str.length);

        trace("=== Done ===");
    }
}
