enum Color {
    Red;
    Green;
    Blue;
}

enum Option {
    Some(v:Int);
    None;
}

class Main {
    static function main() {
        // Test 1: Exhaustive enum switch — no warning expected
        var c = Color.Red;
        switch (c) {
            case Red: trace("red");
            case Green: trace("green");
            case Blue: trace("blue");
        }

        // Test 2: Non-exhaustive — should warn about missing Blue
        switch (c) {
            case Red: trace("red");
            case Green: trace("green");
        }

        // Test 3: Default makes it exhaustive — no warning expected
        switch (c) {
            case Red: trace("red");
            default: trace("other");
        }

        // Test 4: Parameterized enum, exhaustive — no warning expected
        var o = Option.Some(42);
        switch (o) {
            case Some(v): trace(v);
            case None: trace("none");
        }

        // Test 5: Parameterized enum, non-exhaustive — should warn about missing None
        switch (o) {
            case Some(v): trace(v);
        }

        trace("done");
    }
}
