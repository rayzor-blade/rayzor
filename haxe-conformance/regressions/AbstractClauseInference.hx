import haxe.macro.Context;
import haxe.macro.Expr;
abstract ClauseValue<T>(T) from T {}
class ClauseRecord<T> { public function new() {} }
abstract ClauseChoice<T>(Dynamic) from ClauseRecord<T> from T {}
class ClauseOther<T> { public function new() {} }
class ClausePair<A,B> { public function new() {} }
abstract ClausePairChoice<T>(Dynamic) from ClausePair<Int,T> from T {}
enum ClauseEnum<A,B> { Item(a:A, b:B); }
abstract ClauseEnumChoice<T>(Dynamic) from ClauseEnum<Int,T> from T {}
abstract ClauseShapeChoice<T>(Dynamic) from { key:Int, value:T } from T {}
class AbstractClauseInference {
  public static macro function typeName(e:Expr):Expr {
    var name=haxe.macro.TypeTools.toString(Context.typeof(e));
    return macro $v{name};
  }
  public static macro function sameType(a:Expr,b:Expr):Expr {
    var left=haxe.macro.TypeTools.toString(Context.typeof(a));
    var right=haxe.macro.TypeTools.toString(Context.typeof(b));
    return macro $v{left == right};
  }
  #if !macro
  static function wrap<T>(value:ClauseValue<T>):ClauseValue<T> return value;
  static function record<T>(value:ClauseChoice<T>):ClauseRecord<T> return null;
  static function choose<T>(value:ClauseChoice<T>):ClauseChoice<T> return value;
  static function pair<T>(value:ClausePairChoice<T>):ClausePairChoice<T> return value;
  static function enumChoice<T>(value:ClauseEnumChoice<T>):ClauseEnumChoice<T> return value;
  static function shape<T>(value:ClauseShapeChoice<T>):ClauseShapeChoice<T> return value;
  static function main() {
    if (typeName(wrap(17)) != "ClauseValue<Int>") throw "clause parameter";
    if (typeName(record(new ClauseRecord<Int>())) != "ClauseRecord<Int>") throw "nominal clause parameter";
    if (typeName(choose(new ClauseRecord<Int>())) != "ClauseChoice<Int>") throw "clause priority";
    if (typeName(choose(new ClauseOther<Bool>())) != "ClauseChoice<ClauseOther<Bool>>") throw "nominal distinction";
    if (typeName(pair(new ClausePair<Int,Bool>())) != "ClausePairChoice<Bool>") throw "nested clause parameter";
    if (typeName(pair(new ClausePair<String,Bool>())) != "ClausePairChoice<ClausePair<String, Bool>>") throw "fixed clause parameter";
    var mismatch:ClauseEnum<String,Bool> = Item("text", true);
    if (!sameType(enumChoice(mismatch), (null:ClauseEnumChoice<ClauseEnum<String,Bool>>))) throw "fixed enum clause parameter";
    var shapeMismatch:{key:String,value:Bool} = {key:"text", value:true};
    if (!sameType(shape(shapeMismatch), (null:ClauseShapeChoice<{key:String,value:Bool}>))) throw "fixed structural clause parameter";
    var value=wrap(17);
    if (Std.string(value) != "17") throw "clause value";
    Sys.println("CONFORMANCE_OK");
  }
  #end
}
