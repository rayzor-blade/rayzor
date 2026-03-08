class Main {
    static function myFunc(x:Int):Int {
        return x * 2;
    }

    static function otherFunc(x:Int):Int {
        return x * 3;
    }

    static function main() {
        // Test 1: Same function reference
        var f1 = myFunc;
        var f2 = myFunc;
        trace(Reflect.compareMethods(f1, f2));  // true

        // Test 2: Different functions
        var f3 = otherFunc;
        trace(Reflect.compareMethods(f1, f3));  // false

        // Test 3: Same lambda stored twice
        var add = (a:Int, b:Int) -> a + b;
        var addCopy = add;
        trace(Reflect.compareMethods(add, addCopy));  // true

        // Test 4: Different lambdas with same body
        var mul1 = (x:Int) -> x * 2;
        var mul2 = (x:Int) -> x * 2;
        trace(Reflect.compareMethods(mul1, mul2));  // false (different closures)

        // Test 5: null comparisons
        trace(Reflect.compareMethods(null, null));  // true
        trace(Reflect.compareMethods(f1, null));    // false

        trace("done");
    }
}
