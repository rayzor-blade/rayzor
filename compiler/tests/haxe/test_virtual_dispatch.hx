class Base {
    public var value:Int;

    public function new(v:Int) {
        this.value = v;
    }

    public function greet():String {
        return "Base";
    }

    public function getValue():Int {
        return this.value;
    }
}

class Child extends Base {
    override public function greet():String {
        return "Child";
    }

    public function new(v:Int) {
        super(v);
    }
}

class GrandChild extends Child {
    override public function greet():String {
        return "GrandChild";
    }

    public function new(v:Int) {
        super(v);
    }
}

class Main {
    static function callGreet(b:Base):String {
        return b.greet();
    }

    static function main() {
        // Test 1: Direct calls
        var base = new Base(1);
        var child = new Child(2);
        var grand = new GrandChild(3);

        trace(base.greet());     // Base
        trace(child.greet());    // Child
        trace(grand.greet());    // GrandChild

        // Test 2: Polymorphic dispatch through Base type
        trace(callGreet(base));   // Base
        trace(callGreet(child));  // Child
        trace(callGreet(grand));  // GrandChild

        // Test 3: Non-virtual method (getValue not overridden)
        trace(base.getValue());   // 1
        trace(child.getValue());  // 2
        trace(grand.getValue());  // 3

        trace("done");
    }
}
