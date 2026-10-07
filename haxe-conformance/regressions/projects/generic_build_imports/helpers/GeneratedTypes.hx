package helpers;

#if !macro
@:genericBuild(helpers.GeneratedTypes.ChoiceBuild.pick())
class Chosen<Rest> {}
#end

#if macro
import haxe.macro.Context as BuildContext;
import haxe.macro.Expr;
using haxe.macro.TypeTools;

class ChoiceBuild {
    public static function pick():ComplexType {
        var params = switch BuildContext.getLocalType() {
            case TInst(_, params): params;
            case _: throw "expected an instance type";
        };
        var element = params[0].toComplexType();
        return macro:Array<$element>;
    }
}
#end
