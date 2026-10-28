class Leaf extends Branch {
    public var leaf:Int;
    public function new(value:Int) {
        super(value);
        leaf = value + 2;
    }
}
