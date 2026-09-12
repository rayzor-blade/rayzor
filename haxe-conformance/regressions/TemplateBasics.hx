// haxe.Template end to end: text, `::var::`, `::if::`. Its source is a
// battery of the shapes that used to lose the value between hops -- an
// untyped `new List()` bound from its first `add`, `l.first()` on it, a
// `Null<anon>` field read, a return type taken from the first return site,
// an untyped parameter forwarded to a typed one, and `import haxe.ds.List`
// resolving to the class the top-level typedef also names.
class TemplateBasics {
    static function main() {
        if (new haxe.Template("plain text").execute({}) != "plain text") throw "plain";
        if (new haxe.Template("HI ::foo::").execute({foo: "there"}) != "HI there") throw "var";
        if (new haxe.Template("::if x::yes::else::no::end::").execute({x: true}) != "yes") throw "if";
        var ctx = {name: "Joan"};
        if (new haxe.Template("My name is ::name::.").execute(ctx) != "My name is Joan.") throw "local context";
        trace("CONFORMANCE_OK");
    }
}
