// `try` is an expression whose value comes from a catch when the try body
// only throws; a return inside a try still types the enclosing function; a
// constructor without `super()` runs the ancestors' declaration-site
// defaults; an element of an Array<Dynamic> is a box.
private class Ghost { public var value:String; public function new(value) this.value = value; }
private class A { public var ghost = new Ghost("booh!"); public var n = 5; }
private class B extends A {}
private class C extends B { public function new() {} }
private class PA { public function new() {} }
private class PB extends PA {}
class TryExpressionsAndInheritedInits {
    static var sarr:Array<Dynamic> = [0, "Hello", " World", 0.4];
    static function check(label:String, got:String, want:String) {
        if (got != want) throw label + ": " + got + " != " + want;
    }
    static function throwBool(b:Bool) return try { throw b; } catch (b:Bool) "ok:" + b;
    static function throwInt(i:Int) return try { throw i; } catch (i:Int) "ok:" + i;
    static function viaLocal(s:String) { var r = try { throw s; } catch (e:String) "ok:" + e; return r; }
    static function inferred(a:PA) { try { throw a; } catch (e:PA) { return "pa"; } }
    static function hierarchy(a:PA) { try { throw a; } catch (b:PB) { return "first"; } catch (a:PA) { return "second"; } }
    static function main() {
        check("try expression", throwBool(true) + " " + throwBool(false) + " " + throwInt(12) + " " + throwInt(-12) + " " + throwInt(0) + " " + viaLocal("x"), "ok:true ok:false ok:12 ok:-12 ok:0 ok:x");
        check("return inside try", inferred(new PA()) + " " + hierarchy(new PB()) + " " + hierarchy(new PA()), "pa first second");
        var c = new C();
        check("inherited defaults", c.n + " " + c.ghost.value, "5 booh!");
        var arr:Array<Dynamic> = [0, "Hello", " World", 0.4];
        var d:Dynamic = arr[1] + arr[2];
        check("dynamic elements", arr[1] + " " + arr[0] + " " + arr[3] + " " + d + " " + (sarr[1] + sarr[2]), "Hello 0 0.4 Hello World Hello World");
        Sys.println("CONFORMANCE_OK");
    }
}
