class AnonNullableStringFields {
    static function create(value:Dynamic):{?value:String} return {value:value};
    static function main() {
        if (create("text").value != "text") throw "boxed String field";
        if (create(null).value != null) throw "null String field";
        var direct:{?value:String} = {value:"direct"};
        if (direct.value != "direct") throw "raw String field";
        trace("CONFORMANCE_OK");
    }
}
