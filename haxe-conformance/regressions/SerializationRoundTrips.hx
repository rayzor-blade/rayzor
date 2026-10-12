class SerializationRoundTrips {
 static function main() {
  var bytes = haxe.io.Bytes.alloc(255);
  for (i in 0...bytes.length) bytes.set(i,i);
  var restored:haxe.io.Bytes = haxe.Unserializer.run(haxe.Serializer.run(bytes));
  if (restored.length != bytes.length) throw "byte length";
  for (i in 0...bytes.length) if (restored.get(i) != i) throw "byte contents";
  var serializer = new haxe.Serializer(); serializer.useCache = true;
  var value = {a:1,b:2,c:"test"};
  serializer.serialize(value); var serialized=serializer.toString();
  var copy:Dynamic = haxe.Unserializer.run(serialized);
  if (copy.a != 1 || copy.b != 2 || copy.c != "test") throw "anonymous fields";
  var bools:{no:Bool,yes:Bool} = haxe.Unserializer.run(haxe.Serializer.run({no:false,yes:true}));
  if (bools.no || !bools.yes) throw "Boolean fields";
  var reflected:{no:Bool,yes:Bool} = cast {};
  Reflect.setField(reflected,"no",false); Reflect.setField(reflected,"yes",true);
  if (reflected.no || !reflected.yes) throw "reflected Boolean fields";
  serializer.reset();
  if (serializer.toString() != "" || !serializer.useCache) throw "reset settings";
  serializer.serialize(value);
  if (serializer.toString() != serialized) throw "reset references";
  serializer.reset(); serializer.serialize([value,value]);
  var shared:Array<Dynamic> = cast haxe.Unserializer.run(serializer.toString());
  if (shared[0] != shared[1] || shared[0].a != 1) throw "shared reference";
  Sys.println("CONFORMANCE_OK");
 }
}
