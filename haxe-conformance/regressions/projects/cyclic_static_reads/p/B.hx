package p;

import p.A;

enum Mode { Fast; Slow; }

class B {
    public var a:A;
    public static var total:Int = 10;
    public function new() {}
    public function mode(m:Mode):String return m == Fast ? "fast" : "slow";
    public function size():Int return a == null ? 0 : A.count + total;
    public function zero():Void { A.count = 0; }
}
