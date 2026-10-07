package unit;

class ClosureSource {
    var value:Int;
    public function new(value) this.value = value;
    public function add(a, b) return value + a + b;
    public function first():Int->String {
        var prefix = "first";
        return (number:Int) -> prefix + number;
    }
    public function second():Int->String {
        var prefix = "second";
        return (number:Int) -> prefix + number;
    }
}
