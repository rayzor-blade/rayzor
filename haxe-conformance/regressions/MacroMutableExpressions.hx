import haxe.macro.Expr;
#if macro
import haxe.macro.MacroStringTools;
#end

macro function moduleLiteral() return macro "module";
macro function moduleTyping() {
  var rejected = try {
    haxe.macro.Context.typeExpr(macro var value:Void);
    false;
  } catch (_:haxe.macro.Expr.Error) {
    true;
  }
  return macro $v{rejected};
}

class MacroMutableExpressions {
  static macro function remaining(first:Expr, rest:Array<Expr>) {
    return macro $v{rest.length};
  }
  static macro function collect(rest:Array<Expr>) {
    return macro [$a{rest}];
  }
  static macro function replace(expression:Expr) {
    expression.expr = (macro 42).expr;
    expression.pos = (macro here).pos;
    return expression;
  }
  static macro function formatted(expression:Expr) {
    expression.pos = (macro here).pos;
    return macro $v{MacroStringTools.isFormatExpr(expression)};
  }
  static macro function construct(rest:Array<Expr>) {
    var path:TypePath = {pack:[], name:"String", params:[]};
    var type = TPath(path);
    return macro (new $path($a{rest}) : $type);
  }
  static function main() {
    if (moduleLiteral() != "module") throw "module macro";
    if (!moduleTyping()) throw "module macro typer error";
    if (remaining(0) != 0) throw "empty trailing expression array";
    if (remaining(0, 1, 2) != 2) throw "trailing expression array";
    if (collect().length != 0) throw "zero macro arguments";
    if (collect(1, 2).join(",") != "1,2") throw "expression array splice";
    if (replace(1) != 42) throw "mutable expression definition";
    var value = 7;
    if (!formatted('value$value')) throw "macro-context import";
    if (formatted("plain")) throw "double-quoted expression";
    if (construct("constructed") != "constructed") throw "constructor and type splices";
    Sys.println("CONFORMANCE_OK");
  }
}
