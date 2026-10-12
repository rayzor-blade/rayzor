class ErasedNativeMapValues {
 public function new() {}
 function accept(value:Map<String,String>):Void {
  if (value["foo"] != "bar") throw "reflected map argument";
 }
 static function id<T>(value:T):T return haxe.Unserializer.run(haxe.Serializer.run(value));
 static function main() {
  var ints = new haxe.ds.StringMap<Int>(); ints.set("a", 2); ints.set("b", -465);
  var boxed:Dynamic = ints;
  var dynamicInts:haxe.ds.StringMap<Dynamic> = cast boxed;
  if (dynamicInts.get("b") != -465) throw "erased int read";
  dynamicInts.set("a", 7);
  if (ints.get("a") != 7) throw "erased int write";
  var copy = id(ints);
  if (copy.get("a") != 7 || copy.get("b") != -465) throw "serialized int map";
  var strings:haxe.ds.StringMap<String> = cast haxe.Unserializer.run("by3:fooy3:barh");
  if (strings.get("foo") != "bar") throw "deserialized string map";
  var receiver = new ErasedNativeMapValues();
  Reflect.callMethod(receiver, Reflect.field(receiver, "accept"), [strings]);
  var literal:Map<String,Int> = ["zero" => 0, "minus" => -3];
  var literalCopy = id(literal);
  if (literalCopy["zero"] != 0 || literalCopy["minus"] != -3) throw "serialized map literal";
  var values = new haxe.ds.IntMap<Dynamic>(); values.set(1, 1.5); values.set(2, true);
  var restored = id(values);
  if (restored.get(1) != 1.5 || restored.get(2) != true) throw "mixed erased map";
  for (v in [haxe.Int64.make(0,1), haxe.Int64.make(1,-1), haxe.Int64.make(-1,0x7fffffff)]) {
   if (id(v) != v) throw "generic Int64 return";
  }
  Sys.println("CONFORMANCE_OK");
 }
}
