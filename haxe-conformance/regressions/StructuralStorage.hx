private enum Item { Number(value:Int); }
private class Record {
    public var before = 123;
    public var text = "class";
    public var count = 7;
    public var fraction = 1.25;
    public var enabled = true;
    public var item = Number(8);
    public var nullable:Null<Int> = null;
    public var values = [4, 5];
    public function new() {}
}
private typedef View = {
    var text:String;
    var count:Int;
    var fraction:Float;
    var enabled:Bool;
    var item:Item;
    var nullable:Null<Int>;
    var values:Array<Int>;
}
class StructuralStorage {
    static function read(view:View) {
        if (view.text != "class" || view.count != 7) throw "scalar read";
        if (view.fraction != 1.25 || !view.enabled) throw "primitive read";
        if (Std.string(view.item) != "Number(8)") throw "enum read";
        if (view.nullable != null || view.values[1] != 5) throw "reference read";
    }
    static function write(view:View) {
        view.text = "written";
        view.count += 2;
        view.fraction = 2.5;
        view.enabled = false;
        view.item = Number(9);
        view.nullable = 12;
        view.values = [6, 7];
    }
    static function narrow(view:{text:String}) {
        if (view.text != "wide") throw "width read";
        view.text = "narrow";
    }
    static function main() {
        var record = new Record();
        var objects:Array<View> = [record];
        read(objects[0]);
        write(objects[0]);
        if (record.text != "written" || record.count != 9) throw "write identity";
        if (record.fraction != 2.5 || record.enabled) throw "primitive write";
        if (Std.string(record.item) != "Number(9)") throw "enum write";
        if (record.nullable != 12 || record.values[0] != 6) throw "reference write";
        record.text = "changed";
        if (objects[0].text != "changed") throw "read identity";
        var wide = {before:42, text:"wide"};
        var narrowed:Array<{text:String}> = [wide];
        narrow(narrowed[0]);
        if (wide.text != "narrow" || wide.before != 42) throw "width write identity";
        Sys.println("CONFORMANCE_OK");
    }
}
