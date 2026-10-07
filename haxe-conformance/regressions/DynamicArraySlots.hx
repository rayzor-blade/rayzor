class ArrayPoint {
    public var value:Int;
    public function new(value:Int) this.value = value;
}

class DynamicArraySlots {
    static function same<T>(a:T, b:T):Bool return a == b;
    static function main() {
        var numbers:Array<Float> = [-101.5];
        var erased:Dynamic = numbers;
        if (erased[0] != -101.5) throw "float read";
        if (!same(erased[0], -101.5)) throw "generic float read";
        erased[0] %= 100;
        if (numbers[0] != -1.5 || erased[0] != -1.5) throw "float modulo write";
        erased[0] += 2.25;
        if (numbers[0] != 0.75) throw "float addition write";
        var old = erased[0]++;
        if (old != 0.75 || numbers[0] != 1.75) throw "float increment write";
        erased[1] = 3.5;
        if (numbers.length != 2 || numbers[1] != 3.5) throw "float growth";
        var integers = [0, 4];
        var ints:Dynamic = integers;
        if (Std.string(ints[0]) != "0" || ints[1] != 4) throw "integer read";
        ints[1] += 3;
        if (integers[1] != 7) throw "integer write";
        var strings = ["hello", "world"];
        var text:Dynamic = strings;
        if (Std.string(text[0]) != "hello") throw "string read";
        text[1] = "updated";
        if (strings[1] != "updated") throw "string write";
        if (Std.string(text) != "[hello,updated]") throw "dynamic string array formatting";
        var emptyStrings:Dynamic = ["", "test"];
        if (Std.string(emptyStrings) != "[,test]") throw "empty string formatting";
        var booleans = [false, true];
        var bools:Dynamic = booleans;
        if (bools[0] != false || bools[1] != true) throw "boolean read";
        bools[0] = true;
        if (!booleans[0]) throw "boolean write";
        var nested = [[2.5]];
        var rows:Dynamic = nested;
        var row:Dynamic = rows[0];
        if (row[0] != 2.5) throw "nested read";
        row[0] = 3.75;
        if (nested[0][0] != 3.75) throw "nested write";
        var objects:Array<ArrayPoint> = [new ArrayPoint(3), null];
        var points:Dynamic = objects;
        if (points[0].value != 3 || points[1] != null) throw "object slots";
        points[0] = new ArrayPoint(7);
        if (objects[0].value != 7) throw "object write";
        var boxed:Array<Dynamic> = [0, 3.5, "x", false, null];
        var mixed:Dynamic = boxed;
        if (Std.string(mixed[0]) != "0" || mixed[1] != 3.5 || mixed[2] != "x" || mixed[3] != false || mixed[4] != null) throw "boxed slots";
        var textNumber:Dynamic = "0";
        if (same(textNumber, 0.0)) throw "generic dynamic string equality";
        var parsed:Dynamic = haxe.Json.parse('[0,2.5,"text",false,null]');
        if (Std.string(parsed[0]) != "0" || parsed[1] != 2.5 || parsed[2] != "text" || parsed[3] != false || parsed[4] != null) throw "json slots";
        parsed[1] += 1.5;
        parsed[2] += "!";
        if (parsed[1] != 4.0 || parsed[2] != "text!") throw "json writes";
        Sys.println("CONFORMANCE_OK");
    }
}
