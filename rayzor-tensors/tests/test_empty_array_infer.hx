// Regression: an UNTYPED empty array literal `var a = []` binds its element type
// from the first `a.push(e)` / `a[i] = e` (the monomorph rewrite), instead of
// staying Array<Dynamic> and truncating floats (wasm→0, native→f64-bits-as-int).
// Keeps numeric arrays unboxed: Array<Float> routes through the f64 push/get
// path, not the generic one. Values checked native + wasm.

import rayzor.ds.Tensor;
import rayzor.ds.DType;

private class Box { public var v:Int; public function new(v) { this.v = v; } }

class Main {
    static function fillF(a:Array<Float>):Void { a[0] = 0.0078125; }
    static function main() {
        // push Float into untyped empty → binds Array<Float>
        var f = []; f.push(0.0078125); f.push(2.5);
        // index-assign Float into untyped empty (binds via a forward-declared var)
        var d = 0.25; var g = []; g.push(0.0); g[0] = d;
        // push Int into untyped empty stays correct (binds Array<Int>)
        var i = []; i.push(7); i.push(9);
        // covariant pass: untyped empty written through an Array<Float> param
        var c = []; c.push(0.0); fillF(c);
        // computed (binary) float push: maxAbs/127 style
        var mx = 1.0; var q = []; q.push(mx / 8.0);
        // method-call return type (lower-fallback): getFlat returns Float
        var t = Tensor.full([4], 0.5, DType.F32); var m = []; for (k in 0...4) m.push(t.getFlat(k));
        // NOT Float-only: String + a custom class bind too
        var s = []; s.push("a"); s.push("b");
        var p = []; p.push(new Box(7)); p.push(new Box(9));

        var ok = f[0] == 0.0078125 && f[1] == 2.5
            && g[0] == 0.25
            && i[0] == 7 && i[1] == 9
            && c[0] == 0.0078125
            && q[0] == 0.125
            && m[0] == 0.5
            && s[0] == "a" && s[1] == "b"
            && p[0].v == 7 && p[1].v == 9;
        if (ok) {
            Sys.println("PASS empty-array-infer f0=" + f[0] + " g0=" + g[0]
                + " i=" + i[0] + " c0=" + c[0] + " q0=" + q[0]
                + " getFlat=" + m[0] + " str=" + s[0] + " pt=" + p[0].v);
        } else {
            Sys.println("FAIL f0=" + f[0] + " g0=" + g[0] + " i0=" + i[0]
                + " c0=" + c[0] + " q0=" + q[0] + " m0=" + m[0] + " s0=" + s[0]);
        }
    }
}
