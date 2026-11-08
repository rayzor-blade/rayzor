package p;

class Tree {
    public var count:Int;
    public var ptr:Int;
    public var scroll:Scroll;

    public function new(count:Int, ptr:Int) {
        this.count = count;
        this.ptr = ptr;
    }

    public function scrolled():Int {
        var s = scroll;
        return s.x + s.y;
    }
}
