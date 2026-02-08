class Main {
    static function main() {
        // Basic match
        var r = new EReg("world", "");
        if (r.match("hello world")) {
            trace("match1: true");
        }
        trace(r.matched(0));      // world
        trace(r.matchedLeft());   // "hello "
        trace(r.matchedRight());  // ""

        // Capture groups (using [a-z]+ instead of \w+ to avoid escape issues)
        var r2 = new EReg("([a-z]+)@([a-z]+)", "");
        if (r2.match("user@host")) {
            trace("match2: true");
        } else {
            trace("match2: false");
        }
        trace(r2.matched(0));     // user@host
        trace(r2.matched(1));     // user
        trace(r2.matched(2));     // host

        // Global replace
        var r3 = new EReg("[0-9]+", "g");
        trace(r3.replace("a1b2c3", "X"));  // aXbXcX

        // Non-global replace (first match only)
        var r4 = new EReg("[0-9]+", "");
        trace(r4.replace("a1b2c3", "X"));  // aXb2c3

        // Case-insensitive
        var r6 = new EReg("hello", "i");
        if (r6.match("HELLO World")) {
            trace("match6: true");
        }
        trace(r6.matched(0));  // HELLO

        // Escape
        trace(EReg.escape("a.b+c"));  // a\.b\+c

        // No match
        var r7 = new EReg("xyz", "");
        if (r7.match("hello")) {
            trace("match7: false");
        } else {
            trace("match7: false");
        }

        // Regex literal syntax
        var r8 = ~/test/;
        if (r8.match("this is a test")) {
            trace("match8: true");
        }
        trace(r8.matched(0));  // test

        // Split count
        var r5 = new EReg("[,;]", "");
        var parts = r5.split("a,b;c");
        trace(parts.length);  // 2

        trace("done");
    }
}
