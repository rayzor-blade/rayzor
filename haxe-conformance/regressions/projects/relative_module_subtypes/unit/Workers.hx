package unit;

interface Parent {
    function value():Int;
}

interface Child extends Parent {}
interface Other {}

class Worker implements Child {
    public function new() {}
    public function value():Int return 42;
}

class GenericChecks {
    public function new() {}
    public function same<T>(left:T, right:T):Bool return left == right;
    public function identity<T>(value:T):T return value;
}
