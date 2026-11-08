package p;

class Bounds {
    public var x:Int;
    public var y:Int;
    public var ptr:Int;

    public function new(x:Int, y:Int) {
        this.x = x;
        this.y = y;
        ptr = x + y;
    }
}

class Slot {
    public var a:Int = 0;
    public var b:Int = 0;
    public var c:Int = 0;
    public var d:Int = 0;
    public var ptr:Int = 9;
    public var x:Int = 7;

    public function new() {}
}

class Marker extends Slot {
    public var id:Int = 3;
}
