package unit;

class FieldSource {
    public var value:Int;
    @:keep static var Marker = 1;

    public function new() {
        value = 2;
    }

    public function read():Int {
        return value;
    }
}
