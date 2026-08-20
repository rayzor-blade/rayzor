@:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class Holder { public var r:Res; public function new(r:Res) { this.r = r; } }
class c30_borrow_out_of_field {
    static function peek(@:borrow r:Res):Int { return r.v; }
    static function main():Void {
        var h = new Holder(new Res(1));
        // EXPECT SILENT — the control for c29. Borrowing takes nothing away,
        // so the refusal there must not extend to here.
        Sys.println(peek(h.r) + peek(h.r));
    }
}
