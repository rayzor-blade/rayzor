class EmptyNativeStrings {
 static function check(value:Bool):Void { if (!value) throw "empty native string"; }
 static function main() {
  for (value in [haxe.io.Bytes.alloc(0).toString(), haxe.io.Bytes.alloc(0).toHex()]) {
   check(value == "" && value != null);
   var object:{text:String} = {text:value};
   check(object.text == "" && object.text != null);
   var held:Dynamic = object;
   check(held.text == "" && held.text != null);
   check(Reflect.field(object, "text") == "");
   var reflected:{text:String} = cast {};
   Reflect.setField(reflected, "text", value);
   check(reflected.text == "" && reflected.text != null);
  }
  var key="RAYZOR_EMPTY_STRING_REGRESSION";
  Sys.putEnv(key,haxe.io.Bytes.alloc(0).toString());
  if(Sys.getEnv(key)==null) throw "empty removed variable";
  var result:{value:String}={value:Sys.getEnv(key)};
  if(result.value!="" || result.value==null) throw "empty environment field";
  Sys.putEnv(key,null);
  if(Sys.getEnv(key)!=null) throw "null did not remove variable";
  trace("CONFORMANCE_OK");
 }
}
