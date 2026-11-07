package p;

import p.B;

class A {
    public var b:B;
    public static var count:Int = 0;
    public function new() { count++; }
    public function link(other:B):Void { b = other; other.a = this; }
    public function peers():Int return b == null ? 0 : B.total + count;
}
