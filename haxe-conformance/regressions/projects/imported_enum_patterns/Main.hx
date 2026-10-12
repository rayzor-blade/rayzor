import Patterns.UnicodeString;
class Main {
    static var values:Array<UnicodeString> = Patterns.filtered;
    static function main() {
        if (values.length != 2) throw "imported enum pattern filter";
        if (!values[0].match(Only([2])) || !values[1].match(Pair([3], [4]))) throw "enum array patterns";
        if (!~/abc/.match("abc")) throw "EReg method";
        Sys.println("CONFORMANCE_OK");
    }
}
