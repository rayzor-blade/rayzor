import haxe.ds.StringMap;
import ForeignModule.ForeignModuleValue;

class Main {
    static function main() {
        trace(read(ForeignModule.make(2048)));
        trace(readOr(ForeignModule.make(4096)));

        var map = new StringMap<ForeignModuleValue>();
        map.set("x", ForeignModule.make(8192));
        trace(read(map.get("x")));
        trace(readOr(map.get("x")));
    }

    static function read(v:ForeignModuleValue):Int {
        return switch (v) {
            case U32(x): x;
            case _: -1;
        };
    }

    static function readOr(v:ForeignModuleValue):Int {
        return switch (v) {
            case U8(x) | U32(x): x;
            case _: -1;
        };
    }
}
