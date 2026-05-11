// Behavioural test: Reflect / Type RTTI on class instances (declared
// + inherited fields), and `Type.typeof` / `Reflect.isObject` /
// `Type.getSuperClass` chain.

class Animal {
    public var name: String;
    public var legs: Int;
    public function new(n: String, l: Int) {
        this.name = n;
        this.legs = l;
    }
}

class Dog extends Animal {
    public var breed: String;
    public function new(n: String, b: String) {
        super(n, 4);
        this.breed = b;
    }
}

class Main {
    static function main() {
        var d = new Dog("Rex", "Labrador");

        // Type.getClass + getClassName: class identity.
        var cls = Type.getClass(d);
        trace(Type.getClassName(cls));               // Dog

        // Type.getInstanceFields: includes inherited fields.
        trace(Type.getInstanceFields(cls));          // [name, legs, breed]

        // Reflect.field on a declared field.
        trace(Reflect.field(d, "breed"));            // Labrador

        // Reflect.field on an inherited field (walks the hierarchy).
        trace(Reflect.field(d, "name"));             // Rex

        // Reflect.setField then read back via direct access.
        Reflect.setField(d, "breed", "Golden Retriever");
        trace(d.breed);                              // Golden Retriever

        // Reflect.fields(instance): same as Type.getInstanceFields(cls).
        trace(Reflect.fields(d));                    // [name, legs, breed]

        // Reflect.hasField: present + missing.
        trace(Reflect.hasField(d, "breed"));         // true
        trace(Reflect.hasField(d, "banana"));        // false

        // Type.resolveClass round-trip.
        var byName = Type.resolveClass("Dog");
        if (byName == null) {
            trace("resolveClass: null");
        } else {
            trace(Type.getClassName(byName));        // Dog
        }

        // Type.typeof on a class instance returns TClass(<actual>).
        trace(Type.typeof(d));                       // TClass(Dog)

        // Reflect.isObject must recognise class instances.
        trace(Reflect.isObject(d));                  // true
        trace(Reflect.isObject(42));                 // false

        // Type.getSuperClass + getClassName chain.
        var sup = Type.getSuperClass(cls);
        if (sup == null) {
            trace("super: null");
        } else {
            trace(Type.getClassName(sup));           // Animal
        }
    }
}
