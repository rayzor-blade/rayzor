import haxe.macro.Context;
import Type.ValueType;

enum ReflectedChoice {
    Empty;
    Number(value:Int);
}

class EnumReflection {
    public function new() {}
    static function check(value:Dynamic, expected:ValueType) {
        if (!Type.enumEq(Type.typeof(value), expected)) throw "enum identity";
    }
    static function main() {
        if (Type.getEnumName(ValueType) != "ValueType") throw "enum declaration name";
        check(null, TNull);
        check(12, TInt);
        check(1.5, TFloat);
        check(false, TBool);
        check("text", TClass(String));
        check([1, 2], TClass(Array));
        check({field:1}, TObject);
        check(function() {}, TFunction);
        check(Class, TObject);
        check(Enum, TObject);
        check(EnumReflection, TObject);
        check(ReflectedChoice, TObject);
        check(new EnumReflection(), TClass(EnumReflection));
        check(Empty, TEnum(ReflectedChoice));
        check(Number(12), TEnum(ReflectedChoice));
        if (!Type.enumEq(Number(12), Number(12))) throw "parameterized enum";
        if (Type.enumEq(Number(12), Number(13))) throw "enum payload";
        var optional:Null<ReflectedChoice> = Empty;
        if (!Type.enumEq(optional, Empty)) throw "nullable enum";
        if (Type.getClass([1, 2]) != Array) throw "array class";
        if (Type.getClass("text") != String) throw "string class";
        var erased:Dynamic = [1, 2];
        if (Type.getClass(erased) != Array) throw "Dynamic array class";
        var empty:Array<Int> = null;
        if (Type.getClass(empty) != null) throw "null array class";
        Sys.println("CONFORMANCE_OK");
    }
}
