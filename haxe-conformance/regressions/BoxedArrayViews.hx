private class ArrayCell { public static var tag:Int = 1; public var value:Int; public function new(v:Int) value = v; }
class BoxedArrayViews {
 static function id<T>(value:T):T return haxe.Unserializer.run(haxe.Serializer.run(value));
 static function check<T>(a:Array<T>, wanted:String):Void {
  if (Std.string(a) != wanted) throw "generic array formatting";
 }
 static function main() {
  var source:Array<Dynamic> = [0, 1, 2];
  var erased:Dynamic = source;
  var ints:Array<Int> = cast erased;
  if (ints[0] != 0 || ints[1] != 1) throw "typed reads of boxes";
  ints[0] = 7;
  if (source[0] != 7) throw "typed write alias";
  source[1] = 8;
  if (ints[1] != 8) throw "dynamic write alias";
  ints.push(9);
  if (source[3] != 9) throw "typed push alias";
  if (ints.pop() != 9 || source.length != 3) throw "typed pop alias";
  if (!ints.contains(8) || ints.indexOf(8) != 1) throw "typed search";
  if (ints.join(",") != "7,8,2") throw "typed join";
  ints.unshift(6); if (source.shift() != 6) throw "typed unshift alias";
  ints.insert(1, 10); if (!ints.remove(10) || source[1] != 8) throw "typed insert/remove alias";
  var f:Array<Dynamic> = [1.5, -2.5]; var fs:Array<Float> = cast f;
  if (fs[0] != 1.5 || fs[1] != -2.5) throw "float boxed read";
  fs[1] = 4.25; if (f[1] != 4.25) throw "float boxed write";
  var promoted:Array<Float> = cast source;
  if (promoted[0] != 7.0 || promoted[1] != 8.0) throw "boxed integer promotion";
  var inferred = new Array();
  inferred[3] = 0;
  inferred[0] = 0;
  inferred[0] |= 128;
  inferred[3] ^= 7;
  if (inferred[0] != 128 || inferred[3] != 7) throw "inferred array bitwise writes";
  if (haxe.crypto.Md5.encode("") != "d41d8cd98f00b204e9800998ecf8427e") throw "inferred MD5 buffer";
  var b:Array<Dynamic> = [true, false]; var bs:Array<Bool> = cast b;
  if (!bs[0] || bs[1]) throw "bool boxed read";
  bs[1] = true; if (b[1] != true) throw "bool boxed write";
  var s:Array<Dynamic> = ["a", "b"]; var ss:Array<String> = cast s;
  if (ss[0] != "a") throw "string boxed read";
  ss[0] = "c"; if (s[0] != "c") throw "string boxed write";
  var cell = new ArrayCell(12); var objects:Array<Dynamic> = [cell];
  var cells:Array<ArrayCell> = cast objects;
  if (cells[0] != cell || cells[0].value != 12) throw "class boxed read";
  if (Reflect.fields(ArrayCell).indexOf("tag") < 0) throw "class token fields";
  var copied:Array<Int> = cast source.copy();
  var sliced:Array<Int> = cast source.slice(1);
  if (copied[0] != 7 || sliced[0] != 8) throw "copied layout";
  var roundtrip:Array<Int> = id([0, 1, 2]);
  if (roundtrip[0] != 0 || roundtrip[1] != 1) throw "serialized typed array";
  var nullable:Array<Null<Int>> = id([0, null, 2]);
  if (nullable[0] != 0 || nullable[1] != null || nullable[2] != 2) throw "nullable typed array";
  check(roundtrip, "[0,1,2]");
  check(fs, "[1.5,4.25]");
  if (fs.pop() != 4.25 || f.length != 1) throw "float boxed pop";
  var empty:Array<Float> = []; if (empty.pop() != null) throw "empty nullable float pop";
  Sys.println("CONFORMANCE_OK");
 }
}
