// A class used as a value must carry the same runtime id its object headers
// carry — that id is the key the RTTI registry is built under. When the class
// literal lowered to a raw SymbolId instead, every Type API call taking a
// Class<T> literal missed the registry and returned null, while the same query
// through Type.getClass(instance) worked because it reads the header.

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
    static var failures = 0;

    static function check(what:String, got:String, want:String) {
        if (got != want) {
            Sys.println("FAIL " + what + " -- expected " + want + " but got " + got);
            failures++;
        }
    }

    static function main() {
        // Class literal, directly
        check("getClassName(Animal)", Type.getClassName(Animal), "Animal");
        check("getClassName(Dog)", Type.getClassName(Dog), "Dog");

        // Same answer via the object header, which is the id the literal must agree with
        var dog = new Dog("Rex");
        check("getClassName(getClass(dog))", Type.getClassName(Type.getClass(dog)), "Dog");

        // Superclass resolution also takes a class literal
        check("getClassName(getSuperClass(Dog))", Type.getClassName(Type.getSuperClass(Dog)), "Animal");

        Sys.println(failures == 0 ? "PASS type-api" : "FAIL type-api " + failures + " mismatches");
    }
}
