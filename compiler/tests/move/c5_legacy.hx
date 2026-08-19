class Thing { public var v:Int; public function new(v:Int) { this.v = v; } }
class c5_legacy {
    static function take(t:Thing):Int { return t.v; }
    static function main():Void {
        var a = new Thing(6);
        var n = take(a);
        var m = take(a);
        var b = a;
        Sys.println(n + m + b.v);   // EXPECT SILENT: no @:safety anywhere
    }
}
