@:move
class Res {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
    public function get():Int { return v; }
}
class Holder { public var r:Res; public function new(r:Res) { this.r = r; } }
class c31_field_method_receiver {
    static function main():Void {
        var h = new Holder(new Res(1));
        // EXPECT SILENT: a receiver is borrowed, so calling through a field is
        // the supported way to use one.
        Sys.println(h.r.get() + h.r.get() + h.r.v);
    }
}
