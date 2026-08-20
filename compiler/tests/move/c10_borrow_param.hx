@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c10_borrow_param {
    static function peek(@:borrow r:Res):Int { return r.v; }
    static function eat(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(1);
        var x = peek(a);
        var y = peek(a);        // EXPECT SILENT: the signature says it borrows
        var z = eat(a);
        Sys.println(x + y + z);
    }
}
