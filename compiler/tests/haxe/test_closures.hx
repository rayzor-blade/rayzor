class Main {
    static function main() {
        // Test 1: Simple closure without captures (function literal with typed params)
        var add = function(a:Int, b:Int):Int { return a + b; };
        trace(add(3, 4));  // 7

        // Test 2: Closure with capture
        var x = 10;
        var addX = function(a:Int):Int { return a + x; };
        trace(addX(5));  // 15
    }
}
