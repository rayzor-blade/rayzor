@:move
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
@:move
class Handle { var c:Cell; public function new(c:Cell) { this.c = c; } }
class c33_owner_read_while_held {
    static function main():Void {
        var c = new Cell(1);
        var h = new Handle(c);
        Sys.println(c.v);   // EXPECT ERROR: c is held by h
    }
}
