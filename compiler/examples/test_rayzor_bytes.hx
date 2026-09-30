/*
 * Test rayzor.Bytes native byte buffer
 * Tests basic allocation, get/set, and string conversion.
 */

import rayzor.Bytes;

class Main {
    static function main() {
        trace("=== Testing rayzor.Bytes ===");

        // Test 1: Basic allocation and get/set
        trace("--- Test 1: alloc/set/get ---");
        var bytes:Bytes = Bytes.alloc(10);
        trace("Allocated 10 bytes");

        bytes.set(0, 72);   // 'H'
        bytes.set(1, 101);  // 'e'
        bytes.set(2, 108);  // 'l'
        bytes.set(3, 108);  // 'l'
        bytes.set(4, 111);  // 'o'
        trace("Set Hello bytes");

        // Test 2: ofString and toString
        trace("--- Test 2: ofString/toString ---");
        var strBytes:Bytes = Bytes.ofString("Hello World");
        var str = strBytes.toString();
        trace("toString: " + str);

        // Test 3: fill
        trace("--- Test 3: fill ---");
        var fillBytes:Bytes = Bytes.alloc(5);
        fillBytes.fill(0, 5, 65);  // Fill with 'A'
        trace("fill result: " + fillBytes.toString());

        // Test 4: blit
        trace("--- Test 4: blit ---");
        var src:Bytes = Bytes.ofString("COPY");
        var dst:Bytes = Bytes.alloc(10);
        dst.fill(0, 10, 45);  // Fill with '-'
        src.blitTo(0, dst, 3, 4);  // Copy "COPY" to position 3
        trace("blit result: " + dst.toString());

        // Test 5: setInt32/getInt32
        trace("--- Test 5: setInt32/getInt32 ---");
        var intBytes:Bytes = Bytes.alloc(16);
        intBytes.setInt32(0, 12345);
        trace("setInt32 done");

        trace("=== All rayzor.Bytes tests completed! ===");
    }
}
