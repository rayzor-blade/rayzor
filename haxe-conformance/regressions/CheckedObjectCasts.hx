interface EmptyCastInterface {}
interface OtherCastInterface {}
class CheckedObjectCasts implements EmptyCastInterface {
 public function new() {}
 static var count=0;
 static function make():Dynamic { count++; return new CheckedObjectCasts(); }
 static function main() {
  var v = new CheckedObjectCasts();
  var e = cast(v, EmptyCastInterface);
  if (e != v) throw "empty interface identity";
  var c = cast(e, CheckedObjectCasts);
  if (c != v) throw "interface to class";
  var caught=false;
  try cast(e, OtherCastInterface) catch (_:Dynamic) caught=true;
  if (!caught) throw "unrelated empty interface";
  caught=false;
  try cast(v, OtherCheckedClass) catch (_:Dynamic) caught=true;
  if (!caught) throw "unrelated class";
  var d:Dynamic = null;
  if (cast(d, EmptyCastInterface) != null) throw "null cast";
  var dynamicInterface:Dynamic = e;
  if (cast(dynamicInterface, CheckedObjectCasts) != v) throw "boxed interface cast";
  if (cast(make(), CheckedObjectCasts) == null || count != 1) throw "single evaluation";
  var invalid:Array<Dynamic> = [5, 1.5, true, "text", [1,2]];
  for (scalar in invalid) {
   caught=false;
   try cast(scalar, CheckedObjectCasts) catch (_:Dynamic) caught=true;
   if (!caught) throw "nonclass Dynamic cast";
  }
  Sys.println("CONFORMANCE_OK");
 }
}
class OtherCheckedClass { public function new(){} }
