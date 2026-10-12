private interface GenericValue<T> { function get():T; }
private class Value<T> implements GenericValue<T> {
 public var value:T;
 public function new(value:T) this.value=value;
 public function get():T return value;
}
@:multiType(T) private abstract Selected<T>(GenericValue<T>) {
 public function new(value:T);
 @:to static function ofInt(_:GenericValue<Int>, value:Int):Value<Int> return new Value(value);
 @:to static function ofFloat(_:GenericValue<Float>, value:Float):Value<Float> return new Value(value);
 @:to static function ofString(_:GenericValue<String>, value:String):Value<String> return new Value(value);
 @:to function toValue():T return this.get();
}
class MultiTypeGenericConstructors {
 static function main() {
  var a = new Selected(10); var ai:Int = a;
  if (ai != 10) throw "inferred Int";
  var b = new Selected("hello"); var bs:String = b;
  if (bs != "hello") throw "inferred String";
  var c = new Selected<Int>(12); var ci:Int = c;
  if (ci != 12) throw "explicit Int";
  var d:Selected<String> = new Selected("world"); var ds:String = d;
  if (ds != "world") throw "expected String";
  var e = new Selected(1.5); var ef:Float = e;
  if (ef != 1.5) throw "inferred Float";
  var count = 2;
  var f = new Selected<Float>(count); var ff:Float = f;
  if (ff != 2.0) throw "explicit Float promotion";
  Sys.println("CONFORMANCE_OK");
 }
}
