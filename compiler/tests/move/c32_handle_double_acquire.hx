@:move
class Cell { public var v:Int; public function new(v:Int) { this.v = v; } }
@:move
class Handle { var c:Cell; public function new(c:Cell) { this.c = c; } }
class c32_handle_double_acquire {
    static function main():Void {
        var c = new Cell(1);
        var h1 = new Handle(c);
        var h2 = new Handle(c);   // EXPECT ERROR: c is already held exclusively
        Sys.println(c.v);
    }
}
