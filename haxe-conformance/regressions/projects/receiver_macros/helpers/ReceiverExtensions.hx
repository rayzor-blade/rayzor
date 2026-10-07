package helpers;
import haxe.macro.Expr;
class ReceiverExtensions {
    macro public static function dupe(s:String):Expr return macro $v{s + s};
    macro public static function increment(e:Expr):Expr return macro (function() return $e + 1)();
    macro public static function incrementArrow(e:Expr):Expr return macro (() -> $e + 2)();
    macro public static function toUpperCase(e:Expr):Expr return macro "wrong";
}
