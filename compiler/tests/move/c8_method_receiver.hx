@:safety @:move
class Res {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
    public function get():Int { return v; }
}
class c8_method_receiver {
    static function main():Void {
        var a = new Res(1);
        var x = a.get();
        var y = a.get();        // EXPECT SILENT: a receiver is observed, not consumed
        Sys.println(x + y);
    }
}
