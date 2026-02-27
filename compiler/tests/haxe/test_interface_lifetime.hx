interface I {
    public function value():Int;
}

class C implements I {
    var x:Int;

    public function new(v:Int) {
        x = v;
    }

    public function value():Int {
        return x;
    }
}

class Main {
    static function main() {
        var c = new C(1);
        {
            var i:I = c;
            c = new C(2);
            trace(i.value()); // 1
        }
        trace(c.value()); // 2

        var c2 = new C(3);
        var j:I = c2;
        j = new C(4);
        trace(c2.value()); // 3
        trace(j.value()); // 4
    }
}
