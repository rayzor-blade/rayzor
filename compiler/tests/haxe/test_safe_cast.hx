class Animal {
    public var name:String;
    public function new(n:String) {
        name = n;
    }
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
        // Test 1: Downcast Animal → Dog (success - object is actually Dog)
        var animal:Animal = new Dog("Rex", "Shepherd");
        var dog = cast(animal, Dog);
        if (dog != null) {
            trace(dog.breed);  // Shepherd
        } else {
            trace("fail1");
        }

        // Test 2: Downcast Animal → Cat (fail - object is Dog)
        var cat = cast(animal, Cat);
        if (cat != null) {
            trace("fail2");
        } else {
            trace("null");     // null
        }

        // Test 3: Downcast Animal → Cat (success - object is actually Cat)
        var animal2:Animal = new Cat("Whiskers");
        var cat2 = cast(animal2, Cat);
        if (cat2 != null) {
            trace(cat2.name);  // Whiskers
        } else {
            trace("fail3");
        }

        trace("done");
    }
}
