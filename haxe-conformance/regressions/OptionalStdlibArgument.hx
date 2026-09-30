class OptionalStdlibArgument {
    static function find(?start:Int):Int {
        return "Hello world !!!".indexOf(" ", start);
    }

    static function findLast(?start:Int):Int {
        return "a b a".lastIndexOf("a", start);
    }

    static function main() {
        if (find(6) != 11) throw "optional start index";
        if (findLast(3) != 0) throw "optional last start index";

        var raw = "Hello world !!!";
        var cursor = 0;
        inline function advance(char:String = " ", ?start:Int) {
            cursor = raw.indexOf(char, start);
            return cursor = (cursor > -1 ? cursor : raw.length) + char.length;
        }
        var hello = raw.substring(0, advance());
        var world = raw.substring(cursor, advance(cursor));
        if (hello != "Hello " || world != "world ") throw "inline optional argument";

        Sys.println("CONFORMANCE_OK");
    }
}
