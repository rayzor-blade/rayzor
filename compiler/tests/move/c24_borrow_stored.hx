@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class Holder { public var kept:Res; public function new(r:Res) { this.kept = r; }
    public function stash(@:borrow r:Res):Void { this.kept = r; }   // EXPECT ERROR
}
class e4_store_field {
    static function main():Void {
        var a = new Res(1);
        var h = new Holder(new Res(0));
        h.stash(a);
        Sys.println(h.kept.v);
    }
}
