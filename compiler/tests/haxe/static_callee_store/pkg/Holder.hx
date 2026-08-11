package pkg;

import rayzor.Bytes;

class Holder {
    static var direct:Bytes = null;  // assigned WITHOUT passing the static to a call
    static var viaCall:Bytes = null; // assigned FROM a call that takes the static

    static function grow(cur:Bytes, n:Int):Bytes {
        if (cur != null && cur.length >= n) return cur;
        return Bytes.alloc(n);
    }

    static function bumpDirect():Void {
        direct = Bytes.alloc(4096);
        Sys.println("  bumpDirect():  direct==null ? " + (direct == null));
    }

    static function bumpViaCall():Void {
        viaCall = grow(viaCall, 4096);
        Sys.println("  bumpViaCall(): viaCall==null ? " + (viaCall == null));
    }

    public static function entry(doBump:Bool):Void {
        Sys.println("  entry(): direct==null ? " + (direct == null)
            + "   viaCall==null ? " + (viaCall == null));
        if (doBump) { bumpDirect(); bumpViaCall(); }
    }

    public static function directIsNull():Bool { return direct == null; }
    public static function viaCallIsNull():Bool { return viaCall == null; }
}
