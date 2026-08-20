@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c11_borrow_after_move {
    static function peek(@:borrow r:Res):Int { return r.v; }
    static function eat(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var z = eat(a);
        var x = peek(a);        // EXPECT ERROR: a borrow of a moved value is still a use
        Sys.println(x + z);
    }
}
