class RuntimeBooleanResults {
    static function inverted(value:Bool):Bool return !value;
    static function main() {
        var r = new EReg("b", "");
        if (!r.match("aba")) throw "negated match";
        if (!!r.match("aaa")) throw "double negated non-match";
        if (inverted(r.match("aba"))) throw "boolean argument";
        var results = [r.match("aba"), r.match("aaa")];
        if (!results[0] || results[1]) throw "boolean array";
        if (!StringTools.startsWith("abc", "a")) throw "startsWith";
        if (StringTools.endsWith("abc", "x")) throw "endsWith";
        if (!sys.FileSystem.exists(".")) throw "exists";
        Sys.println("CONFORMANCE_OK");
    }
}
