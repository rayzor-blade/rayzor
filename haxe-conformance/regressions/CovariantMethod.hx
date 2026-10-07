class CovariantParent {
    public function new() {}
    public function values():Dynamic return [9];
}

class CovariantChild extends CovariantParent {
    public function new() super();
    public function length():Int return values().length;
    override public function values():Array<Int> return [1, 2, 3];
}

class CovariantMethod {
    static function main() {
        var child = new CovariantChild();
        if (child.length() != 3) throw "covariant result type";
        Sys.println("CONFORMANCE_OK");
    }
}
