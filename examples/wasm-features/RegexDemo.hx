/**
 * EReg demo — regex via browser RegExp / native PCRE.
 *
 * Run: rayzor run --wasm examples/wasm-features/RegexDemo.hx
 */
class RegexDemo {
    static function main() {
        trace("=== Regex Demo ===");

        // Basic match
        var r = ~/([0-9]+)-([a-z]+)/;
        if (r.match("item-42-hello-world")) {
            trace("Full match: " + r.matched(0));
            trace("Group 1: " + r.matched(1));
            trace("Group 2: " + r.matched(2));
        }

        // Left/right of match
        var r2 = ~/world/;
        if (r2.match("hello world foo")) {
            trace("Left of 'world': '" + r2.matchedLeft() + "'");
            trace("Right of 'world': '" + r2.matchedRight() + "'");
        }

        // Replace
        var r3 = ~/[aeiou]/g;
        var replaced = r3.replace("hello world", "*");
        trace("Replace vowels: " + replaced);

        // Escape
        var escaped = EReg.escape("hello.world[0]");
        trace("Escaped: " + escaped);

        trace("=== Done ===");
    }
}
