// A method call on a Dynamic that several runtime classes could answer is
// bound at run time by the box's tag; a value none of them claims throws
// where it is reached rather than failing the module. The receiver comes
// out of its box for the class's own function, and the result goes back
// into one. A function held in a Dynamic is called through its box, and a
// `return` inside a `try` keeps its value (the try's exit branch replaced
// it, and the function fell off its end with none).
class DynamicDispatch {
    static function inTry():Int { try { return 5; } catch (e:Dynamic) { return -1; } }
    static function inTryDyn():Dynamic { try { return [1, 2]; } catch (e:Dynamic) { throw "x"; } }
    static function main() {
        var a:Dynamic = [1, 2];
        if (a.join(",") != "1,2") throw "join on a Dynamic array";
        a.push(3);
        if (a.length != 3) throw "push on a Dynamic array";
        var p:Dynamic = a.pop();
        if (p != 3) throw "pop on a Dynamic array";
        var it:Dynamic = a.iterator();
        if (it == null || it.hasNext == null) throw "iterator on a Dynamic array";
        var v:Iterator<Dynamic> = it;
        var n = 0; var s = "";
        for (x in v) { n++; s += Std.string(x) + ","; }
        if (n != 2 || s != "1,2,") throw "for over a Dynamic iterator: " + s;
        var m:Dynamic = new Map<String, Int>();
        m.set("k", 5);
        if (!m.exists("k") || m.get("k") != 5) throw "map through Dynamic";
        var str:Dynamic = "abc";
        if (str.toUpperCase() != "ABC") throw "string method through Dynamic";
        var threw = false;
        try { var i:Dynamic = 5; i.iterator(); } catch (e:Dynamic) { threw = true; }
        if (!threw) throw "unclaimed dispatch must throw";
        var f:Dynamic = function() { return 5; };
        if (f() != 5) throw "call through a Dynamic function box";
        if (inTry() != 5) throw "return inside try";
        var r:Dynamic = inTryDyn();
        if (r.length != 2) throw "Dynamic return inside try";
        var t = new haxe.Template("::if (n > 2)::big::else::small::end::");
        if (t.execute({n: 5}) != "big") throw "template comparison";
        if (t.execute({n: 1}) != "small") throw "template comparison false";
        var fe = new haxe.Template("::foreach items::[::__current__::]::end::");
        if (fe.execute({items: [1, 2]}) != "[1][2]") throw "template foreach";
        trace("CONFORMANCE_OK");
    }
}
