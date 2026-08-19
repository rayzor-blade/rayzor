@:safety @:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c4_reassign {
    static function take(r:Res):Int { return r.v; }
    static function main():Void {
        var a = new Res(4);
        var n = take(a);
        a = new Res(5);            // revived
        var m = take(a);           // EXPECT SILENT
        Sys.println(n + m);
    }
}
