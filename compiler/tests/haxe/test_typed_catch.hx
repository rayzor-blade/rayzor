class MyError {
    public var msg:String;
    public function new(m:String) {
        this.msg = m;
    }
}

class Main {
    static function main() {
        // Test 1: catch Int
        try {
            throw 42;
        }
        catch (e:String) {
            trace("WRONG: string");
        }
        catch (e:Int) {
            trace(e); // Should print 42
        }
        catch (e:Dynamic) {
            trace("WRONG: dynamic");
        }

        // Test 2: catch String
        try {
            throw "hello";
        }
        catch (e:Int) {
            trace("WRONG: int");
        }
        catch (e:String) {
            trace(e); // Should print hello
        }
        catch (e:Dynamic) {
            trace("WRONG: dynamic");
        }

        // Test 3: Dynamic catches anything
        try {
            throw 99;
        }
        catch (e:Dynamic) {
            trace(e); // Should print 99
        }

        // Test 4: catch class instance
        try {
            throw new MyError("boom");
        }
        catch (e:Int) {
            trace("WRONG: int");
        }
        catch (e:MyError) {
            trace(e.msg); // Should print boom
        }
        catch (e:Dynamic) {
            trace("WRONG: dynamic");
        }

        // Test 5: Dynamic catches class too
        try {
            throw new MyError("fallback");
        }
        catch (e:Dynamic) {
            trace("caught dynamic"); // Should print caught dynamic
        }
    }
}
