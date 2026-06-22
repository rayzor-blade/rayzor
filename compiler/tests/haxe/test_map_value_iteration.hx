// Regression: `for (v in map)` must iterate the map's VALUES, not its keys.
//
// In Haxe, `for (x in map)` iterates values (keys come from `map.keys()`).
// Two bugs made this wrong:
//   1) hir_to_mir lowered single-variable map for-in to keys_to_array + index,
//      binding the loop variable to each KEY. So `for (v in intMap) sum += v`
//      summed the keys.
//   2) ast_lowering typed the loop variable as the KEY type. For a String-keyed
//      map that made `sum += v` compile as string concat and cast the Int value
//      to a string pointer -> SIGSEGV.
// Fixed by iterating values_to_array and typing the loop variable as the value.

class Main {
    static function main() {
        // Int keys, Int values: sum the values (10+20=30), not the keys (1+2=3).
        var ii = new Map<Int, Int>();
        ii.set(1, 10);
        ii.set(2, 20);
        var iiSum = 0;
        for (v in ii) iiSum += v;

        // String keys, Int values: previously SIGSEGV'd (v mis-typed as String).
        var si = new Map<String, Int>();
        si.set("a", 5);
        si.set("b", 7);
        var siSum = 0;
        for (v in si) siSum += v;

        // Int keys, String values: concatenated string values (order-independent
        // length check, since map iteration order is unspecified).
        var is = new Map<Int, String>();
        is.set(1, "ab");
        is.set(2, "cd");
        var isLen = 0;
        for (v in is) isLen += v.length;

        // k=>v iteration must still bind v to the value and k to the key.
        var kv = new Map<Int, Int>();
        kv.set(3, 30);
        kv.set(4, 40);
        var kvSum = 0;
        for (k => v in kv) kvSum += k + v;

        var ok = iiSum == 30 && siSum == 12 && isLen == 4 && kvSum == 77;
        Sys.println(ok
            ? "PASS"
            : "FAIL ii=" + iiSum + " si=" + siSum + " isLen=" + isLen + " kv=" + kvSum);
    }
}
