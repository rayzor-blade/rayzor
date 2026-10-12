private enum GenericParameter<T> { Payload(value:T); }
private class GenericParent { public function new() {} }
private class GenericChild extends GenericParent { public function new() super(); }

class GenericReferenceIdentity {
 static function box<T>(value:T):Dynamic return value;
 static function relay<T>(value:T):Dynamic return box(value);
 static function text<T>(value:T):String return Std.string(value);
 static function forward<T>(value:T):String return text(value);
 static function main() {
  var absent:Array<Int> = null;
  if (relay(absent) != null || forward(absent) != "null") throw "null array identity";
  var a:Array<Int> = [1, 2];
  var erased:Dynamic = relay(a);
  if (!Std.isOfType(erased, Array)) throw "array identity";
  var read:Array<Int> = cast erased;
  read[0] = 3;
  if (a[0] != 3 || read.length != 2) throw "shared array storage";
  if (haxe.Serializer.run(erased) != "ai3i2h") throw "array slots";
  if (forward(a) != "[3,2]") throw "array formatting";
  var floats:Array<Float> = [1.5, -2.5];
  if (forward(floats) != "[1.5,-2.5]") throw "float array layout";
  var nullableFloats:Null<Array<Float>> = floats;
  if (forward(nullableFloats) != "[1.5,-2.5]") throw "nullable array layout";
  var nested:Array<Array<Int>> = [[4,5]];
  if (forward(nested) != "[[4,5]]") throw "nested array layout";
  var enums:Array<GenericParameter<Int>> = [Payload(7)];
  if (forward(enums) != "[Payload(7)]") throw "enum array layout";
  var d = Date.fromTime(1234567890000.0);
  var boxedDate:Dynamic = relay(d);
  if (!Std.isOfType(boxedDate, Date)) throw "date identity";
  if (Type.getClass(boxedDate) != Date) throw "native class identity";
  var nullableDate:Null<Date> = d;
  if (!Std.isOfType(relay(nullableDate), Date)) throw "nullable class identity";
  var missingDate:Null<Date> = null;
  if (relay(missingDate) != null) throw "nullable class null";
  var date:Date = cast boxedDate;
  if (date.getTime() != d.getTime()) throw "date storage";
  var integer = haxe.Int64.make(1, -1);
  var boxedInteger:Dynamic = relay(integer);
  if (Type.getClass(boxedInteger) == null || Type.getClass(boxedInteger) != Type.getClass((integer:Dynamic))) throw "Int64 identity";
  var restored:haxe.Int64 = cast boxedInteger;
  if (restored != integer || forward(integer) != Std.string((integer:Dynamic))) throw "Int64 storage";
  if (forward([integer]) != "[" + Std.string((integer:Dynamic)) + "]") throw "Int64 array layout";
  var values = new haxe.ds.StringMap<Int>(); values.set("a", 7);
  var boxedMap:Dynamic = relay(values);
  if (!Std.isOfType(boxedMap, haxe.ds.StringMap)) throw "map identity";
  var map:haxe.ds.StringMap<Int> = cast boxedMap;
  if (map.get("a") != 7) throw "map storage";
  var child:GenericParent = new GenericChild();
  var boxedChild:Dynamic = relay(child);
  if (Type.getClass(boxedChild) != GenericChild || !Std.isOfType(boxedChild, GenericChild)) throw "subclass identity";
  Sys.println("CONFORMANCE_OK");
 }
}
