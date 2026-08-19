@:move class Res {
    public var id:Int;
    public function new(i:Int) { this.id = i; }
}

class C {
    static function sink(r:Res):Int { return r.id; }

    static function main() {
        var a = new Res(1);
        var i = 0;
        var n = 3;
        var s = 0;
        while (i < n) {
            s += sink(a);
            i++;
        }
        trace(s);
    }
}
