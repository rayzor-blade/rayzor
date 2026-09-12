// `var l = new List(); l.add(x)` resolves l to List<typeof x> on that first
// use, the way Haxe resolves a monomorph. Until then the local carries the
// class's own formal T, and an erased `first()` on it can never be unboxed --
// haxe.Template's parseBlock is written exactly this way, and returned a box
// that its enum switch could not match.
enum TE { OpStr(s:String); OpBlock(l:List<TE>); }
class GenericLocalFromUse {
    var buf:StringBuf;
    function new() {}
    function run(e:TE) { switch (e) { case OpStr(s): buf.add(s); case OpBlock(l): for (x in l) run(x); } }
    function exec(e:TE):String { buf = new StringBuf(); run(e); return buf.toString(); }
    // an un-annotated callee declared BELOW its caller, returning enum ctors
    function build():TE { var l = new List(); l.add(mk("a")); l.add(mk("b")); return OpBlock(l); }
    function mk(s:String) { return OpStr(s); }
    static function main() {
        var t = new GenericLocalFromUse();
        var l = new List();                 // no type argument
        l.add(OpStr("solo"));
        var e:TE = l.first();
        if (t.exec(e) != "solo") throw "untyped local List, first()";
        if (t.exec(t.build()) != "ab") throw "forward enum-returning callee";
        var li = new List();
        li.add(41);
        var v = li.first();
        if (v + 1 != 42) throw "untyped local List over Int, first()";
        trace("CONFORMANCE_OK");
    }
}
