@:derive([PartialEq, PartialOrd, Hash])
class Person {
    public var name:String;
    public var age:Int;

    public function new(name:String, age:Int) {
        this.name = name;
        this.age = age;
    }
}

class Main {
    static function main() {
        var alice = new Person("Alice", 30);
        var alice2 = new Person("Alice", 30);
        var bob = new Person("Bob", 25);

        // PartialEq with string fields
        trace(alice == alice2);  // true (same name + age)
        trace(alice == bob);     // false
        trace(alice != bob);     // true

        // PartialOrd with string fields (lexicographic: name first, then age)
        trace(alice < bob);   // true (Alice < Bob lexicographically)
        trace(bob > alice);   // true

        // Hash
        var h1 = alice.hashCode();
        var h2 = alice2.hashCode();
        trace(h1 == h2);     // true (same fields → same hash)
        trace(h1 != 0);      // true (non-trivial)
    }
}
