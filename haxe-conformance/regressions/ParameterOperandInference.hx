import haxe.ds.Option;
import haxe.macro.Context;
import haxe.macro.Expr;
typedef ConcatPrefix = String;
class ConcatRecord {
  public var value:String;
  public function new(prefix:String,a,b,c) value=prefix+a+b+c;
}
class ParameterOperandInference {
  public static macro function isEnum(expression:Expr):Expr {
    var result=switch Context.typeof(expression) {
      case TEnum(_, _): true;
      default: false;
    };
    return macro $v{result};
  }
  #if !macro
  static function concatAlias(prefix:ConcatPrefix,a,b) return prefix+a+b;
  static function concat(prefix:String,a,b,c) return prefix+a+b+c;
  static function mixed(prefix:String,a:Int,b:Bool,c:Float) return prefix+a+b+c;
  static function literal(a,b,c) return ""+a+b+c;
  static function getNone(t = None) return t;
  static function shadow(prefix:String, value) {
    { var prefix:Int=1; return prefix+value; }
  }
  static function main() {
    if (shadow("ignored",17) != 18) throw "shadowed string prefix";
    if (!isEnum(getNone())) throw "enum default type";
    var local=function(t=None) return t;
    if (!isEnum(local())) throw "local enum default type";
    if ("enum:" + local() != "enum:None") throw "local enum default";
    var arrow=(prefix:String,a,b) -> prefix+a+b;
    if (arrow("prefix:","a","b") != "prefix:ab") throw "arrow prefix";
    if (concatAlias("prefix:","a","b") != "prefix:ab") throw "aliased prefix";
    if ("enum:" + getNone() != "enum:None") throw "enum default";
    if (concat("prefix:", "a", "b", "c") != "prefix:abc") throw "typed prefix";
    if (literal("a", "b", "c") != "abc") throw "literal prefix";
    if (mixed("prefix:", 1, false, 1.5) != "prefix:1false1.5") throw "mixed operands";
    var record=Type.createInstance(ConcatRecord,["prefix:", "a", "b", "c"]);
    if (record.value != "prefix:abc") throw "reflective concatenation";
    Sys.println("CONFORMANCE_OK");
  }
  #end
}
