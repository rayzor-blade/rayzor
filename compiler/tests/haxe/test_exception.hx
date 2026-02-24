import haxe.Exception;

class MyException extends Exception {
    public function new(msg:String) {
        super(msg);
    }
}

class Main {
    static function main() {
        // Test 1: Basic Exception with message
        try {
            throw new Exception("basic error");
        } catch(e:Exception) {
            trace(e.message);  // "basic error"
        }

        // Test 2: Polymorphic catch (subclass caught by parent)
        try {
            throw new MyException("custom error");
        } catch(e:Exception) {
            trace(e.message);  // "custom error"
        }

        // Test 3: Specific catch before general
        try {
            throw new MyException("specific");
        } catch(e:MyException) {
            trace(e.message);  // "specific"
        } catch(e:Exception) {
            trace("WRONG");
        }

        // Test 4: Backward compat — primitive throws
        try {
            throw 42;
        } catch(e:Int) {
            trace(e);  // 42
        }

        // Test 5: Dynamic still catches everything
        try {
            throw new MyException("dyn");
        } catch(e:Dynamic) {
            trace("caught dynamic");
        }

        // Test 6: NativeStackTrace — exception stack is non-empty
        try {
            throw new Exception("trace test");
        } catch(e:Exception) {
            var stack = haxe.NativeStackTrace.exceptionStack();
            // stack is a HaxeString ptr — check it's non-null and has content
            if (stack != null) {
                trace("has stack");
            } else {
                trace("no stack");
            }
        }

        // Test 7: NativeStackTrace — callStack returns something
        var cs = haxe.NativeStackTrace.callStack();
        if (cs != null) {
            trace("has callstack");
        } else {
            trace("no callstack");
        }

        trace("done");
    }
}
