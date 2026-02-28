class Animal {
    var name:String;
    public function new(n:String) {
        this.name = n;
    }
}

class Dog extends Animal {
    public function new(n:String) {
        super(n);
    }
}

class Main {
    static function main() {
        var className = Type.getClassName(Animal);
        trace(className);  // Animal

        var dogName = Type.getClassName(Dog);
        trace(dogName);    // Dog

        // Test Type.getClass() - reads object header type_id
        var dog = new Dog("Rex");
        var cls = Type.getClass(dog);
        var name = Type.getClassName(cls);
        trace(name);       // Dog
    }
}
