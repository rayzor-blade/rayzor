import haxe.ds.StringMap;

class Main {
    static function main() {
        trace(read(ForeignPayloadEnum.U32(2048)));
        trace(readOr(ForeignPayloadEnum.U32(4096)));

        var map = new StringMap<ForeignPayloadEnum>();
        map.set("x", ForeignPayloadEnum.U32(8192));
        trace(read(map.get("x")));
        trace(readOr(map.get("x")));
    }

    static function read(v:ForeignPayloadEnum):Int {
        return switch (v) {
            case U32(x): x;
            case _: -1;
        };
    }

    static function readOr(v:ForeignPayloadEnum):Int {
        return switch (v) {
            case U8(x) | U32(x): x;
            case _: -1;
        };
    }
}
