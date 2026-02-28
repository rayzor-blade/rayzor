class Animal {
    public var name:String;
    public function new(n:String) { name = n; }
}

class Dog extends Animal {
    public var breed:String;
    public function new(n:String, b:String) {
        super(n);
        breed = b;
    }
}

class Cat extends Animal {
    public function new(n:String) {
        super(n);
    }
}

class Main {
    static function main() {
        // Test 1: is operator with class hierarchy
        var d:Animal = new Dog("Rex", "Labrador");
        trace(d is Animal);  // true (static type match)
        trace(d is Dog);     // true (downcast)

        var c = new Cat("Whiskers");
        trace(c is Animal);  // true (upcast)
        trace(c is Cat);     // true (same type)

        // Test 2: is operator with Dynamic primitives
        var x:Dynamic = 42;
        trace(x is Int);     // true

        var s:Dynamic = "hello";
        trace(s is String);  // true

        var b:Dynamic = true;
        trace(b is Bool);    // true

        var f:Dynamic = 3.14;
        trace(f is Float);   // true

        // Test 3: Std.isOfType() method calls
        trace(Std.isOfType(d, Animal));  // true
        trace(Std.isOfType(d, Dog));     // true

        // Test 4: Cross-type checks (should be false)
        trace(42 is String);  // false
        trace(42 is Float);   // false
        trace(42 is Bool);    // false

        // Test 5: Negative class checks
        var dog = new Dog("Buddy", "Poodle");
        trace(dog is Cat);    // false (unrelated classes)

        trace("done");
    }
}
