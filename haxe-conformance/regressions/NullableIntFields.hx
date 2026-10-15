class NullableIntFields {
    static function result(code:Null<Int>):{code:Int, text:String} {
        return {code:code, text:"result"};
    }
    static function main() {
        var zero = result(0);
        var negative = result(-7);
        if (zero.code != 0 || zero.text != "result") throw "zero result";
        if (negative.code != -7) throw "negative result";
        if (Reflect.field(negative, "code") != -7) throw "reflected result";
        var value:Null<Int> = 12;
        var typed:{code:Int} = {code:value};
        if (typed.code != 12 || Reflect.field(typed, "code") != 12) throw "typed field";
        var nullable:{code:Null<Int>} = {code:value};
        if (nullable.code != 12) throw "nullable field";
        trace("CONFORMANCE_OK");
    }
}
