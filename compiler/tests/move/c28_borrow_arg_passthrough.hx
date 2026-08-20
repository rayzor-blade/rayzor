@:move
class Res { public var v:Int; public function new(v:Int) { this.v = v; } }
class c28_borrow_arg_passthrough {
    static function peek(@:borrow r:Res):Int { return r.v; }
    // Returning the RESULT of a call that borrowed is not an escape — a value
    // came back, not the reference: EXPECT SILENT.
    static function twice(@:borrow r:Res):Int { return peek(r) + peek(r); }
    static function main():Void { Sys.println(twice(new Res(1))); }
}
