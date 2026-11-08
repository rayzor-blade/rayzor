package p;

class Scroll {
    public var tree:Tree;
    public var x:Int;
    public var y:Int;

    public function new(tree:Tree, x:Int, y:Int) {
        this.tree = tree;
        this.x = x;
        this.y = y;
    }

    public function owner():Int {
        var t = tree;
        return t.ptr * 100 + t.count;
    }
}
