interface IGreeter {
    public function greet():String;
}

class Greeter implements IGreeter {
    public function new() {}

    public function greet():String {
        return "hi";
    }
}

class Other {
    public function new() {}
}

class Main {
    static function main() {
        var dynGood:Dynamic = new Greeter();
        var dynBad:Dynamic = new Other();
        var typedGood = new Greeter();

        trace(Std.is(dynGood, IGreeter)); // true
        trace(Std.is(dynBad, IGreeter));  // false
        trace(Std.is(typedGood, IGreeter)); // true
    }
}
