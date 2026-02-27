interface IValue {
    public function value():Int;
}

class ValueBox implements IValue {
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
        var c = new ValueBox(10);
        var i:IValue = c;
        var j:IValue = i;

        i = new ValueBox(20);
        trace(j.value()); // 10
        trace(i.value()); // 20

        var k:IValue = i;
        i = new ValueBox(30);
        trace(k.value()); // 20
        trace(i.value()); // 30
    }
}
