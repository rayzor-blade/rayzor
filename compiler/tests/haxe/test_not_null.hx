class Main {
    // Test @:notNull annotation and null flow analysis

    static function safeLength(@:notNull s:String):Int {
        // s is guaranteed non-null here, safe to access directly
        return s.length;
    }

    static function main() {
        // Test 1: @:notNull parameter works
        trace(safeLength("hello")); // 5

        // Test 2: Null check narrowing
        var x:String = null;
        if (x != null) {
            trace("not null: " + x);
        } else {
            trace("is null"); // expected
        }

        // Test 3: Non-null assignment
        var y:String = "world";
        trace(y.length); // 5

        // Test 4: Conditional null narrowing
        var z:String = null;
        z = "assigned";
        if (z != null) {
            trace("z=" + z); // z=assigned
        }

        trace("done");
    }
}
