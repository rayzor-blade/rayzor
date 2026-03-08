class Main {
    static function add(a:Int, b:Int):Int {
        return a + b;
    }

    static function mul3(a:Int, b:Int, c:Int):Int {
        return a * b * c;
    }

    static function greet(name:String, greeting:String):String {
        return greeting + ", " + name + "!";
    }

    static function apply(f:Int->Int, x:Int):Int {
        return f(x);
    }

    static function main() {
        // Test 1: Bind first arg, placeholder second
        var add5 = add.bind(5, _);
        trace(add5(3)); // 8

        // Test 2: Bind all args (zero-param function)
        var seven = add.bind(3, 4);
        trace(seven()); // 7

        // Test 3: Bind second arg, placeholder first
        var addTen = add.bind(_, 10);
        trace(addTen(7)); // 17

        // Test 4: Multiple placeholders (identity-like)
        var addAlias = add.bind(_, _);
        trace(addAlias(11, 22)); // 33

        // Test 5: 3-arg function, bind first only
        var mul3by2 = mul3.bind(2, _, _);
        trace(mul3by2(3, 4)); // 24

        // Test 6: 3-arg function, bind middle
        var mulX5Z = mul3.bind(_, 5, _);
        trace(mulX5Z(2, 3)); // 30

        // Test 7: String function bind
        var hello = greet.bind(_, "Hello");
        trace(hello("World")); // Hello, World!

        // Test 8: Pass bound function to higher-order function
        var inc = add.bind(1, _);
        trace(apply(inc, 99)); // 100

        trace("done");
    }
}
