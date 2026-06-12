import haxe.ds.StringMap;

enum Metaish {
    U8(v:Int);
    I8(v:Int);
    U16(v:Int);
    I16(v:Int);
    U32(v:Int);
    I32(v:Int);
    U64(v:Int);
    I64(v:Int);
    F32(v:Float);
    F64(v:Float);
    Bool(v:Bool);
    Str(s:String);
    Arr(elemType:Int, items:Array<Metaish>);
}

class Main {
    static function main() {
        var x = read(U32(2048));
        trace("u32=" + x);
        var y = readOr(U32(2048));
        trace("u32.or=" + y);
        var z = readForeign(ForeignMetaish.U32(2048));
        trace("foreign.u32.or=" + z);
        trace("foreign.u32=" + readForeignU32(ForeignMetaish.U32(2048)));
        var s = readString(Str("llama"));
        trace("str=" + s);
        var m = new StringMap<Metaish>();
        m.set("x", U32(2048));
        trace("map.u32=" + read(m.get("x")));
        trace("map.u32.or=" + readOr(m.get("x")));
        var fm = new StringMap<ForeignMetaish>();
        fm.set("x", ForeignMetaish.U32(2048));
        trace("map.foreign.u32.or=" + readForeign(fm.get("x")));
        trace("map.foreign.u32=" + readForeignU32(fm.get("x")));
    }

    static function read(v:Metaish):Int {
        return switch (v) {
            case U32(x): x;
            case _: -1;
        };
    }

    static function readString(v:Metaish):String {
        return switch (v) {
            case Str(s): s;
            case _: "";
        };
    }

    static function readOr(v:Metaish):Int {
        return switch (v) {
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: -1;
        };
    }

    static function readForeign(v:ForeignMetaish):Int {
        return switch (v) {
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: -1;
        };
    }

    static function readForeignU32(v:ForeignMetaish):Int {
        return switch (v) {
            case U32(x): x;
            case _: -1;
        };
    }
}
