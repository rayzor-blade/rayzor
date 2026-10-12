private enum ReflectedPayload { Value(n:Int); }
class ReflectedGenericArrays {
 public function new() {}
 public function text<T>(values:Array<T>):String return Std.string(values[0]);
 public function absent<T>(values:Array<T>):Bool return values[0] == null;
 static function main() {
  var instance = new ReflectedGenericArrays();
  var name = "te" + "xt";
  var method = Reflect.field(instance, name);
  var integers:Array<Int> = [7];
  var floats:Array<Float> = [1.5];
  var bools:Array<Bool> = [true];
  if (Reflect.callMethod(instance, method, [integers]) != "7") throw "reflected integer array";
  if (Reflect.callMethod(instance, method, [floats]) != "1.5") throw "reflected float array";
  if (Reflect.callMethod(instance, method, [bools]) != "true") throw "reflected bool array";
  if (Reflect.callMethod(instance, method, [["x"]]) != "x") throw "reflected string array";
  if (Reflect.callMethod(instance, method, [[Value(3)]]) != "Value(3)") throw "reflected enum array";
  var nullName = "ab" + "sent";
  if (Reflect.callMethod(instance, Reflect.field(instance, nullName), [[null]]) != true) throw "reflected generic null";
  Sys.println("CONFORMANCE_OK");
 }
}
