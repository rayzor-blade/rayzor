class DynamicArithmetic {
    static function sum(a:Dynamic, b:Dynamic):Dynamic return a + b;
    static function addInt(a:Dynamic, b:Int):Dynamic return a + b;
    static function intAdd(a:Int, b:Dynamic):Dynamic return a + b;
    static function subtract(a:Dynamic, b:Int):Dynamic return a - b;
    static function multiply(a:Dynamic, b:Int):Dynamic return a * b;
    static function divide(a:Dynamic, b:Int):Dynamic return a / b;
    static function intDivide(a:Int, b:Dynamic):Dynamic return a / b;
    static function modulo(a:Dynamic, b:Int):Dynamic return a % b;
    static function expect(value:Dynamic, expected:Dynamic, label:String) {
        if (value != expected) throw label;
    }
    static function main() {
        expect(sum(2, 3), 5, "Dynamic sum");
        expect(addInt(2, 3), 5, "mixed sum");
        expect(intAdd(2, 3), 5, "reverse mixed sum");
        expect(addInt(-5, 2), -3, "negative mixed sum");
        expect(Std.string(addInt(-5, 2)), "-3", "negative Dynamic representation");
        expect(addInt(1.5, 2), 3.5, "mixed float sum");
        expect(intAdd(2, 1.5), 3.5, "reverse mixed float sum");
        expect(subtract(5.5, 2), 3.5, "mixed float difference");
        expect(multiply(1.5, 3), 4.5, "mixed float product");
        expect(divide(3, 2), 1.5, "mixed division");
        expect(intDivide(3, 2), 1.5, "reverse mixed division");
        expect(modulo(5.5, 2), 1.5, "mixed float modulo");
        expect(addInt("item", 3), "item3", "mixed string sum");
        Sys.println("CONFORMANCE_OK");
    }
}
