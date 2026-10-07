import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;

#if !macro
@:genericBuild(GenericBuildTypes.textType())
private class BuiltText<T> {}
@:genericBuild(GenericBuildTypes.arrayType())
private class BuiltArray<Rest> {}
@:genericBuild(GenericBuildTypes.classType())
private class BuiltClass {}
@:genericBuild(GenericBuildTypes.dynamicType())
private class BuiltDynamic {}
@:genericBuild(GenericBuildTypes.labelType())
private class BuiltLabel<Rest> {}
@:genericBuild(GenericBuildTypes.GenericBuildTypes.textType())
private class BuiltQualifiedText {}
@:genericBuild(GenericBuildTypes.GenericBuildSub.arrayType())
private class BuiltQualifiedArray {}
@:genericBuild(GenericBuildTypes.constructorType())
private class BuiltConstructor<Rest> {}
private class BuildLabel<Const> {}
private class BuildTarget {
    public static function fromCharCode(code:Int):String return String.fromCharCode(code);
}
private class BuildConstructorTarget {
    public var text:String;
    public var number:Int;
    public function new(text:String, number:Int) {
        this.text = text;
        this.number = number;
    }
}
#end

class GenericBuildTypes {
    #if macro
    public static function textType():ComplexType return macro:String;
    public static function classType():ComplexType return macro:BuildTarget;
    public static function dynamicType():ComplexType return macro:Dynamic;
    public static function constructorType():ComplexType {
        var args = Context.getCallArguments();
        if (args == null || args.length != 2) throw "expected constructor arguments";
        if (Context.typeof(args[0]).toString() != "String" || Context.typeof(args[1]).toString() != "Int")
            throw "unexpected constructor types";
        return macro:BuildConstructorTarget;
    }
    public static function arrayType():ComplexType {
        var params = switch Context.getLocalType() {
            case TInst(_, params): params;
            case _: throw "expected an instance type";
        };
        var element = params[0].toComplexType();
        return macro:Array<$element>;
    }
    public static function labelType():ComplexType {
        var label = switch Context.getLocalType() {
            case TInst(_, params): params.map(t -> t.toString()).join("_");
            case _: throw "expected an instance type";
        };
        var ct = TPath({pack:[], name:"BuildLabel", params:[TPExpr(macro $v{label})]});
        return macro:$ct;
    }
    #end
    macro static function checkType(value:Expr, expected:Expr):Expr {
        var actualType = Context.typeof(value).toString();
        var expectedType = Context.typeof(expected).toString();
        if (actualType != expectedType) Context.error(actualType + " != " + expectedType, value.pos);
        return macro null;
    }
    #if !macro
    static function text():BuiltText<Int> return "generated";
    static function main() {
        var s:BuiltText<Float> = text();
        checkType(s, "");
        if (s.toUpperCase() != "GENERATED") throw "type substitution";
        var ints:BuiltArray<Int> = [1, 2];
        var strings:BuiltArray<String> = ["one", "two"];
        checkType(ints, ([]:Array<Int>));
        checkType(strings, ([]:Array<String>));
        if (ints[1] != 2 || strings[1] != "two") throw "per-use parameters";
        var target = BuiltClass;
        if (target.fromCharCode(65) != "A") throw "class value substitution";
        var copied = target;
        final frozen = BuiltClass;
        if (copied.fromCharCode(66) != "B" || frozen.fromCharCode(67) != "C") throw "stored class values";
        var dynamicValue:BuiltDynamic = null;
        checkType(dynamicValue, (null:Dynamic));
        checkType((null:BuiltLabel), (null:BuildLabel<"">));
        checkType((null:BuiltLabel<Int, String>), (null:BuildLabel<"Int_String">));
        var qualifiedText:BuiltQualifiedText = "qualified";
        var qualifiedArray:BuiltQualifiedArray = [3, 4];
        checkType(qualifiedText, "");
        checkType(qualifiedArray, ([]:Array<Int>));
        if (qualifiedText.length != 9 || qualifiedArray[1] != 4) throw "module subtype macro paths";
        var constructed = new BuiltConstructor("constructor", 7);
        if (constructed.text != "constructor" || constructed.number != 7) throw "constructor arguments";
        Sys.println("CONFORMANCE_OK");
    }
    #end
}

#if macro
class GenericBuildSub {
    public static function arrayType():ComplexType return macro:Array<Int>;
}
#end
