private enum RecursiveValue { Node(value:Dynamic); }
class RecursiveStringification {
    static function check(value:Dynamic, label:String) {
        for (_ in 0...2) {
            var text = Std.string(value);
            if (text == null || text.length == 0 || text.length > 4096) throw label;
        }
    }
    static function main() {
        var object = {rec:(null:Dynamic)};
        object.rec = object;
        check(object, "recursive object");
        var array:Array<Dynamic> = [];
        array.push(array);
        check(array, "recursive array");
        var record = {rec:(null:Dynamic)};
        var node = Node(record);
        record.rec = node;
        check(node, "recursive enum");
        var shared = {value:42};
        var text = Std.string(shared);
        var values:Array<Dynamic> = [shared, shared];
        if (Std.string(values) != "[" + text + "," + text + "]")
            throw "shared object reference";
        Sys.println("CONFORMANCE_OK");
    }
}
