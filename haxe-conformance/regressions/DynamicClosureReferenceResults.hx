class DynamicClosureReferenceResults {
    static function main() {
        var source:{function values(input:Array<Int>):Array<Int>;} = new ReferenceResultSource();
        var values = source.values([10, 20, 30]);
        if (values.length != 3 || values[0] != 10 || values[2] != 30) throw "structural method array result";

        var integers:Dynamic = () -> [4, 5];
        var array:Array<Int> = integers();
        if (array.length != 2 || array[1] != 5) throw "dynamic array result";
        var nested:Dynamic = () -> [["left", "right"]];
        var arrays:Array<Array<String>> = nested();
        if (arrays.length != 1 || arrays[0][1] != "right") throw "nested array result";
        if (Std.string(nested()) != "[[left,right]]") throw "array slot layout";

        var factory:Dynamic = () -> new ReferenceResultSource();
        var instance:ReferenceResultSource = factory();
        if (instance.values([7])[0] != 7) throw "class result";
        if (Type.getClassName(Type.getClass(factory())) != "ReferenceResultSource") throw "class result tag";
        var functions:Dynamic = () -> ((value:Int) -> value + 2);
        var callback:Dynamic = functions();
        if (!Reflect.isFunction(callback) || callback(3) != 5) throw "function result";
        Sys.println("CONFORMANCE_OK");
    }
}
class ReferenceResultSource {
    public function new() {}
    public function values(input:Array<Int>):Array<Int> return input.copy();
}
