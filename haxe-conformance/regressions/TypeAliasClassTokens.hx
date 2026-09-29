import haxe.ds.List;

class TypeAliasClassTokens {
    static var tokens:Array<Dynamic> = [List, Date];

    static function main() {
        if (Type.getClassName(tokens[0]) != "haxe.ds.List") throw "List type token";
        if (Type.getClassName(tokens[1]) != "Date") throw "Date type token";
        Sys.println("CONFORMANCE_OK");
    }
}
