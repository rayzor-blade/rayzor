class Main {
    static function drive(cb:Int->String->Bool):Bool {
        return cb(7, "hi");
    }

    static function main() {
        var seen:Array<Int> = new Array<Int>();
        var ok = drive(function(id:Int, text:String):Bool {
            seen.push(id);
            return text == "hi" && seen.length == 1 && seen[0] == 7;
        });

        if (!ok) throw "callback returned false";
        if (seen.length != 1) throw "callback not invoked";
        trace(seen[0]);
    }
}
