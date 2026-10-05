// Pidigits benchmark — the Computer Language Benchmarks Game spigot algorithm
// over a small arbitrary-precision integer (base-10000 limbs, little-endian),
// since Haxe has no portable BigInt. Prints a checksum and the last digits.
//
// Tests: Int array arithmetic with carries, allocation of growing arrays

package benchmarks;

class Pidigits {
    static inline var N = 1000;
    static inline var BASE = 10000;

    static function fromInt(v:Int):Array<Int> {
        return v == 0 ? [] : [v];
    }

    static function copy(a:Array<Int>):Array<Int> {
        return a.copy();
    }

    static function trim(a:Array<Int>) {
        var n = a.length;
        while (n > 0 && a[n - 1] == 0)
            n--;
        if (n < a.length)
            a.resize(n);
    }

    // a *= m, for 0 <= m < 200000.
    static function mulSmall(a:Array<Int>, m:Int) {
        if (m == 0) {
            a.resize(0);
            return;
        }
        var carry = 0;
        for (i in 0...a.length) {
            var v = a[i] * m + carry;
            a[i] = v % BASE;
            carry = Std.int(v / BASE);
        }
        while (carry > 0) {
            a.push(carry % BASE);
            carry = Std.int(carry / BASE);
        }
    }

    // a += b.
    static function addInPlace(a:Array<Int>, b:Array<Int>) {
        var carry = 0;
        var n = a.length > b.length ? a.length : b.length;
        for (i in 0...n) {
            var v = (i < a.length ? a[i] : 0) + (i < b.length ? b[i] : 0) + carry;
            if (v >= BASE) {
                v -= BASE;
                carry = 1;
            } else {
                carry = 0;
            }
            if (i < a.length)
                a[i] = v;
            else
                a.push(v);
        }
        if (carry > 0)
            a.push(carry);
    }

    // a -= b, for a >= b.
    static function subInPlace(a:Array<Int>, b:Array<Int>) {
        var borrow = 0;
        for (i in 0...a.length) {
            var v = a[i] - (i < b.length ? b[i] : 0) - borrow;
            if (v < 0) {
                v += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            a[i] = v;
        }
        trim(a);
    }

    // a = b - a, for b >= a.
    static function reverseSubInPlace(a:Array<Int>, b:Array<Int>) {
        var borrow = 0;
        for (i in 0...b.length) {
            var v = b[i] - (i < a.length ? a[i] : 0) - borrow;
            if (v < 0) {
                v += BASE;
                borrow = 1;
            } else {
                borrow = 0;
            }
            if (i < a.length)
                a[i] = v;
            else
                a.push(v);
        }
        trim(a);
    }

    static function compare(a:Array<Int>, b:Array<Int>):Int {
        if (a.length != b.length)
            return a.length < b.length ? -1 : 1;
        var i = a.length - 1;
        while (i >= 0) {
            if (a[i] != b[i])
                return a[i] < b[i] ? -1 : 1;
            i--;
        }
        return 0;
    }

    // q and t stay positive; r goes negative after some eliminations, so it
    // carries its sign beside its magnitude.
    var q:Array<Int>;
    var r:Array<Int>;
    var rNeg:Bool;
    var t:Array<Int>;
    var k:Int;

    function new() {
        q = fromInt(1);
        r = fromInt(0);
        rNeg = false;
        t = fromInt(1);
        k = 0;
    }

    // r += x, for x >= 0.
    function addToR(x:Array<Int>) {
        if (!rNeg) {
            addInPlace(r, x);
        } else if (compare(r, x) > 0) {
            subInPlace(r, x);
        } else {
            reverseSubInPlace(r, x);
            rNeg = false;
        }
    }

    // r -= x, for x >= 0.
    function subFromR(x:Array<Int>) {
        if (rNeg) {
            addInPlace(r, x);
        } else if (compare(r, x) >= 0) {
            subInPlace(r, x);
        } else {
            reverseSubInPlace(r, x);
            rNeg = true;
        }
    }

    // Whether q > r.
    function qExceedsR():Bool {
        return rNeg || compare(q, r) > 0;
    }

    function nextTerm() {
        k++;
        var k2 = k * 2 + 1;
        addToR(q);
        addToR(q);
        mulSmall(r, k2);
        mulSmall(q, k);
        mulSmall(t, k2);
    }

    // floor((q * nth + r) / t), a single decimal digit here; r >= q > 0 when
    // this is called.
    function extract(nth:Int):Int {
        var num = copy(q);
        mulSmall(num, nth);
        addInPlace(num, r);
        var d = 0;
        while (compare(num, t) >= 0) {
            subInPlace(num, t);
            d++;
        }
        return d;
    }

    function eliminate(d:Int) {
        var dt = copy(t);
        mulSmall(dt, d);
        subFromR(dt);
        mulSmall(r, 10);
        mulSmall(q, 10);
    }

    public static function main() {
        var p = new Pidigits();
        var sum = 0;
        var tail = new StringBuf();
        var i = 0;
        while (i < N) {
            p.nextTerm();
            if (p.qExceedsR())
                continue;
            var d = p.extract(3);
            if (d != p.extract(4))
                continue;
            sum = (sum * 10 + d) % 1000003;
            if (i >= N - 10)
                tail.add(d);
            i++;
            p.eliminate(d);
        }
        Sys.println("digits " + N + " checksum " + sum + " last " + tail.toString());
    }
}
