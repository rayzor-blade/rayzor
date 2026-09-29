private enum TokenChoice { First; Second; }

class DynamicTypeTokenEquality {
    static function sameParameter(typeValue:Dynamic, index:Int, typeList:Array<Dynamic>):Bool {
        return typeList[index] == typeValue && typeValue == typeList[index];
    }

    static function main() {
        var tokens:Array<Dynamic> = [Int, String, Array, TokenChoice];
        var intType:Dynamic = Int;
        var stringType:Dynamic = String;
        var arrayType:Dynamic = Array;
        var enumType:Dynamic = TokenChoice;

        if (tokens[0] != intType || intType != tokens[0]) throw "Int token";
        if (tokens[1] != stringType || stringType != tokens[1]) throw "String token";
        if (tokens[2] != arrayType || arrayType != tokens[2]) throw "Array token";
        if (tokens[3] != enumType || enumType != tokens[3]) throw "enum token";
        if (tokens[0] == stringType) throw "different tokens";
        if (!sameParameter(Int, 0, tokens)) throw "Int parameter";
        if (!sameParameter(String, 1, tokens)) throw "String parameter";
        if (!sameParameter(Array, 2, tokens)) throw "Array parameter";
        if (!sameParameter(TokenChoice, 3, tokens)) throw "enum parameter";
        Sys.println("CONFORMANCE_OK");
    }
}
