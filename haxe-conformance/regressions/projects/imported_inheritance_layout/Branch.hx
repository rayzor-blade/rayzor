class Branch extends Root {
    public var branch:Int;
    public function new(value:Int) {
        super(value);
        branch = value + 1;
    }
}
