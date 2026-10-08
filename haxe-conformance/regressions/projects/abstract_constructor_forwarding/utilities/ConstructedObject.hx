package utilities;

class ConstructedObject {
    public static var calls = 0;
    public var number:Int;
    public function new(value:Int) {
        calls++;
        number = value + 7;
    }
}
