class Main {
    static function main() {
        // Test 1: Basic throw and catch
        try {
            trace("before");
            throw 42;
            trace("after");  // should NOT print
        } catch (e:Dynamic) {
            trace(e);  // 42
        }
        trace("done");  // done

        // Test 2: No exception thrown - should skip catch
        try {
            trace("no throw");
        } catch (e:Dynamic) {
            trace("should not print");
        }
        trace("ok");  // ok
    }
}
