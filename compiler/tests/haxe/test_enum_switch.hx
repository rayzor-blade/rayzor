enum Option {
    Some(v:Int);
    None;
}

class Main {
    static function main() {
        var x = Option.Some(42);
        switch (x) {
            case Some(v): trace(v);    // 42
            case None: trace("none");
        }

        var y = Option.None;
        switch (y) {
            case Some(v): trace(v);
            case None: trace("none");  // none
        }

        // Test wildcard/default
        var z = Option.Some(99);
        switch (z) {
            case None: trace("none");
            default: trace("other");   // other
        }
    }
}
