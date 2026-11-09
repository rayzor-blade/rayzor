interface IV { var ptr(default, null):Int; function twice():Int; }
class Col implements IV {
  public var ptr(default, null):Int; public final rgb:Int;
  public function new(rgb:Int) { this.rgb = rgb; this.ptr = rgb + 1; }
  public function twice():Int return rgb * 2;
}
interface IS<T> { function get():T; function set(v:T):Void; }
class SV<T:IV> implements IS<T> {
  var current:T;
  public function new(v:T) current = v;
  public function get():T return current;
  public function set(v:T):Void current = v;
  public function ptrOf():Int return current.ptr;
  public function twiceOf():Int return current.twice();
  public function asIV():IV return current;
}
typedef SIV = { var ptr(default, null):Int; function twice():Int; }
class SS<T:SIV> {
  public var current:T;
  public function new(v:T) current = v;
  public function get():T return current;
  public function ptrOf():Int return current.ptr;
  public function twiceOf():Int return current.twice();
}
class SBField<T:IV> {
  public var current:T;
  public function new(v:T) current = v;
  public function ptrOf():Int return current.ptr;
}
class InterfaceBoundedParams {
  static function check(value:Bool, label:String):Void {
    if (!value)
      throw label;
  }

  static function main():Void {
    var sv = new SV<Col>(new Col(10));
    check(sv.get().rgb == 10, "bounded getter read as the class");
    check(sv.ptrOf() == 11, "property of a bounded value");
    check(sv.twiceOf() == 20, "interface method on a bounded value");
    check(sv.asIV().ptr == 11, "bounded value passed as the interface");
    var i:IS<Col> = sv;
    check(i.get().rgb == 10, "bounded getter through an unbounded interface");
    i.set(new Col(20));
    check(sv.get().rgb == 20 && sv.ptrOf() == 21 && sv.twiceOf() == 40, "set through an unbounded interface");
    var si = new SV<IV>(new Col(30));
    check(si.ptrOf() == 31 && si.twiceOf() == 60, "bounded parameter bound to the interface");
    var f = new SBField<Col>(new Col(65280));
    check(f.current.rgb == 65280, "bounded field read as the class");
    var c:Col = f.current;
    check(c.ptr == 65281, "bounded field assigned to the class");
    f.current = new Col(7);
    check(f.ptrOf() == 8, "bounded field written from outside");
    var ss = new SS<Col>(new Col(40));
    check(ss.get().rgb == 40 && ss.ptrOf() == 41 && ss.twiceOf() == 80, "structural bound");
    trace("CONFORMANCE_OK");
  }
}
