@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c23_borrow_field_read {
    static function peek(@:borrow r:Res):Int { return r.v; }   // EXPECT SILENT: a field READ
    static function main():Void {
        var a = new Res(1);
        Sys.println(peek(a) + peek(a));
    }
}
