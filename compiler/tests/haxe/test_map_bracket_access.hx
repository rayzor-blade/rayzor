// Bracket access on a Map must dispatch to Map.get/Map.set.
//
// It does not. `m[k] = v` lowers to ARRAY indexing, so:
//   - with a real Map behind it, the read returns the wrong value (0, not 1);
//   - with `var m:Map<K,V> = []` behind it, the initialiser builds a HaxeArray
//     rather than a Map, the KEY POINTER becomes the array index, and
//     haxe_array_set_i64 grows the backing store to cover that index and
//     zero-fills it -- gigabytes for a single insert.
//
// Only the `new Map()` form is exercised here. The `[]`-literal form is the
// same defect and is left out ON PURPOSE: it allocates until the OS kills it
// (61 GB on haxe-conformance unit.issues.Issue2479), which is not something a
// test run should do to the machine.
//
// haxe 4.3.6 prints 1 and 2 for the two cases below.
class Main {
    static function main() {
        // Bracket READ off a .set() write -- this half works.
        var a:Map<String, Int> = new Map();
        a.set("a", 1);
        var readOk = (a["a"] == 1);

        // Bracket WRITE -- this is the broken half. The value never reaches the
        // map, so the read that follows returns the default rather than 1.
        var b:Map<String, Int> = new Map();
        b["b"] = 1;
        var writeThenGet = b.get("b");
        var writeThenBracket = b["b"];

        if (readOk && writeThenGet == 1 && writeThenBracket == 1) {
            Sys.println("PASS map-bracket");
        } else {
            Sys.println("FAIL map-bracket read=" + a["a"]
                + " write->get=" + writeThenGet
                + " write->bracket=" + writeThenBracket
                + " (all should be 1; bracket ASSIGNMENT does not reach Map.set)");
        }
    }
}
