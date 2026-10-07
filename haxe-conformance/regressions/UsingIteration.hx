using UsingIteration.ClassLoops;
using UsingIteration.AbstractLoops;
using UsingIteration.OwnLoops;
class ClassLoops {
    public static function iterator(value:ClassValues):Iterator<Int> return [1, 2].iterator();
}
class AbstractLoops {
    public static function iterator(value:AbstractValues):Iterator<Int> return [3, 4].iterator();
}
class OwnLoops {
    public static function iterator(value:DirectValues):Iterator<Int> return [8].iterator();
}
class DirectValues {
    var more = true;
    public function new() {}
    public function hasNext():Bool return more;
    public function next():Int { more = false; return 7; }
}
class ClassValues implements ArrayAccess<Int> {
    public var length:Int = 99;
    public function new() {}
}
abstract AbstractValues(String) from String {
    public var length(get, never):Int;
    function get_length() return this.length;
    @:arrayAccess function get(index:Int) return -index;
}
class UsingIteration {
    static var evaluations = 0;
    static function makeValues():ClassValues { evaluations++; return new ClassValues(); }
    static function check(value:Bool, label:String) { if (!value) throw label; }
    static function main() {
        var classText = "";
        for (n in makeValues()) classText += n;
        check(classText == "12", "class extension iterator");
        check(evaluations == 1, "iterable evaluated once");
        var abstractText = "";
        var values:AbstractValues = "abcd";
        for (n in values) abstractText += n;
        check(abstractText == "34", "abstract extension iterator");
        var generated = [for (n in values) n * 2];
        check(generated.length == 2 && generated[0] == 6 && generated[1] == 8, "extension comprehension");
        var direct = "";
        for (n in new DirectValues()) direct += n;
        check(direct == "7", "direct iterator before extension");
        Sys.println("CONFORMANCE_OK");
    }
}
