class Pair {
    public var x:Int;
    public var y:Int;
    public var label:String;

    public function new(a:Int, b:Int = 0, s:String = "default") {
        x = a;
        y = b;
        label = s;
    }
}

class Main {
    static function add(a:Int, b:Int, c:Int = 100):Int {
        return a + b + c;
    }

    static function greet(name:String, prefix:String = "Hello"):String {
        return prefix + " " + name;
    }

    static function main() {
        // Constructor with all args
        var p1 = new Pair(1, 2, "full");
        trace(p1.x);      // 1
        trace(p1.y);      // 2
        trace(p1.label);   // full

        // Constructor with 2 args (label defaults to "default")
        var p2 = new Pair(10, 20);
        trace(p2.x);      // 10
        trace(p2.y);      // 20
        trace(p2.label);   // default

        // Constructor with 1 arg (y defaults to 0, label to "default")
        var p3 = new Pair(99);
        trace(p3.x);      // 99
        trace(p3.y);      // 0
        trace(p3.label);   // default

        // Static function with all args
        trace(add(1, 2, 3));   // 6

        // Static function with default c=100
        trace(add(1, 2));      // 103

        // Static function with string default
        trace(greet("World"));          // Hello World
        trace(greet("World", "Hi"));    // Hi World

        trace("done");
    }
}
