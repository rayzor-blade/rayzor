@:generic private class ConstantValue<@:const VALUE> {
  public function new() {}
  public function get() return VALUE;
}
@:generic private class ConstantPair<@:const A, @:const B> {
  public function new() {}
  public function get() return A + " " + B;
}
@:generic private class ConstantClosure<@:const N> {
  public function new() {}
  public function get() return function() return N;
  public function local() { var N = 9; return N; }
  public function argument(N:Int) return N;
  public function nested() {
    var x = 0;
    { var N = 8; x = N; }
    return x + cast(N, Int);
  }
}
class ConstantGenericValues {
  static function main() {
    if (new ConstantPair<"foo",12>().get() != "foo 12") throw "constant tuple";
    var value:ConstantValue<1> = new ConstantValue();
    if (value.get() + 2 != 3) throw "constructor context";
    if (new ConstantValue<-3>().get() != -3) throw "negative constant";
    if (new ConstantValue<true>().get() != true) throw "boolean constant";
    if (new ConstantValue<1.5>().get() != 1.5) throw "float constant";
    if (new ConstantValue<"X">().get() != "X" || new ConstantValue<"Y">().get() != "Y") throw "distinct strings";
    var a:EReg = new ConstantValue<~/a/>().get();
    var b:EReg = new ConstantValue<~/b/>().get();
    if (!a.match("a") || a.match("b") || b.match("a") || !b.match("b")) throw "distinct regex constants";
    var c = new ConstantClosure<4>();
    if (c.get()() != 4 || c.local() != 9 || c.argument(7) != 7 || c.nested() != 12) throw "lexical binding";
    var again:ConstantValue<1> = new ConstantValue<1>();
    if (again.get() != value.get()) throw "repeated specialization";
    Sys.println("CONFORMANCE_OK");
  }
}
