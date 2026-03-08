/*
 * Test StringTools - non-inline methods only
 * (inline methods require compiler support for field access during inlining)
 */

class Main {
    static function main() {
        trace("=== Testing StringTools (non-inline methods) ===");

        // Test 1: isSpace (not inline)
        trace("--- Test 1: isSpace ---");
        var result1 = StringTools.isSpace(" abc", 0);
        if (result1) {
            trace("PASS: space at index 0 detected");
        } else {
            trace("FAIL: space should be detected");
        }

        var result2 = StringTools.isSpace("abc", 0);
        if (!result2) {
            trace("PASS: 'a' is not space");
        } else {
            trace("FAIL: 'a' should not be space");
        }

        // Test 2: ltrim (not inline on rayzor)
        trace("--- Test 2: ltrim ---");
        var ltrimmed = StringTools.ltrim("  Hello");
        trace("ltrim result: '" + ltrimmed + "'");
        if (ltrimmed == "Hello") {
            trace("PASS: ltrim works");
        } else {
            trace("FAIL: Expected 'Hello'");
        }

        // Test 3: rtrim (not inline on rayzor)
        trace("--- Test 3: rtrim ---");
        var rtrimmed = StringTools.rtrim("Hello  ");
        trace("rtrim result: '" + rtrimmed + "'");
        if (rtrimmed == "Hello") {
            trace("PASS: rtrim works");
        } else {
            trace("FAIL: Expected 'Hello'");
        }

        // Test 4: trim (not inline on rayzor)
        trace("--- Test 4: trim ---");
        var trimmed = StringTools.trim("  Hello  ");
        trace("trim result: '" + trimmed + "'");
        if (trimmed == "Hello") {
            trace("PASS: trim works");
        } else {
            trace("FAIL: Expected 'Hello'");
        }

        // Test 5: lpad
        trace("--- Test 5: lpad ---");
        var padded = StringTools.lpad("5", "0", 3);
        trace("lpad result: '" + padded + "'");
        if (padded == "005") {
            trace("PASS: lpad works");
        } else {
            trace("FAIL: Expected '005'");
        }

        // Test 6: rpad
        trace("--- Test 6: rpad ---");
        var rpadded = StringTools.rpad("5", "0", 3);
        trace("rpad result: '" + rpadded + "'");
        if (rpadded == "500") {
            trace("PASS: rpad works");
        } else {
            trace("FAIL: Expected '500'");
        }

        // Test 7: replace
        trace("--- Test 7: replace ---");
        var replaced = StringTools.replace("Hello World", "World", "Rayzor");
        trace("replace result: '" + replaced + "'");
        if (replaced == "Hello Rayzor") {
            trace("PASS: replace works");
        } else {
            trace("FAIL: Expected 'Hello Rayzor'");
        }

        // Test 8: hex (2 args, no default fill)
        trace("--- Test 8: hex ---");
        var hexVal = StringTools.hex(255, 2);
        trace("hex(255,2): '" + hexVal + "'");
        if (hexVal == "FF") {
            trace("PASS: hex works");
        } else {
            trace("FAIL: Expected 'FF', got '" + hexVal + "'");
        }

        trace("=== StringTools tests completed! ===");
    }
}
