private abstract TemplateValue(haxe.Template) {
    public inline function get():haxe.Template return this;
    @:from public static inline function fromString(value:String):TemplateValue
        return cast new haxe.Template(value);
    @:to public inline function toString():String
        return this.execute({value:"converted"});
}
class AggregateAbstractConversions {
    static function main() {
        var input = "::value::";
        var object:{template:TemplateValue} = {template:input};
        if (object.template.get().execute({value:"first"}) != "first") throw "object from";
        object = {template:input};
        if (object.template.get().execute({value:"second"}) != "second") throw "assigned object from";
        var nested:{object:{template:TemplateValue}} = {object:{template:input}};
        if (nested.object.template.get().execute({value:"nested"}) != "nested") throw "nested object from";
        var text:{value:String} = {value:object.template};
        if (text.value != "converted") throw "object to";
        text = {value:object.template};
        if (text.value != "converted") throw "assigned object to";
        var templates:Array<TemplateValue> = [input];
        if (templates[0].get().execute({value:"array"}) != "array") throw "array from";
        var strings:Array<String> = [object.template];
        if (strings[0] != "converted") throw "array to";
        Sys.println("CONFORMANCE_OK");
    }
}
