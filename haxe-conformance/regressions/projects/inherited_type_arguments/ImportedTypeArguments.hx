import helper.ArgumentChild;
import helper.ArgumentRenamed;
import helper.ArgumentFixed;
class ImportedTypeArguments {
    static function main() {
        var child = new ArgumentChild<Int>([13]);
        var value = child.value;
        if (!Std.isOfType(value, Array) || value[0] != 13) throw "imported field type";
        var methodValue = child.get();
        if (!Std.isOfType(methodValue, Array) || methodValue[0] != 13) throw "imported method type";
        var renamed = new ArgumentRenamed<Int>([17]);
        var inherited = renamed.value;
        if (!Std.isOfType(inherited, Array) || inherited[0] != 17) throw "imported transitive binding";
        var fixed = new ArgumentFixed("imported");
        var text = fixed.value;
        if (!Std.isOfType(text, String) || text.toUpperCase() != "IMPORTED") throw "imported fixed argument";
        Sys.println("CONFORMANCE_OK");
    }
}
