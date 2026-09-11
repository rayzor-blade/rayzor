// A method's inferred return type must not depend on whether it is declared
// before or after its caller. haxe.Template hit this: its constructor calls
// parseTokens(), declared further down, so the List<Token> decayed to Dynamic
// and `.isEmpty()` matched several native classes instead -- the whole module
// then failed to compile and `Template.new` became a trap stub.
private typedef Tok = { var s:Bool; var p:String; }
class ForwardReturnInference {
    // the caller comes FIRST; both callees are declared below it
    public function new(str:String) {
        var toks = parseToks(str);
        if (toks.isEmpty()) throw "isEmpty on a forward-declared return";
        if (toks.length != 1) throw "length on a forward-declared return";
        var direct = makeDirect();
        if (!direct.isEmpty()) throw "return new C<T>() directly";
    }
    function parseToks(data:String) {
        var toks = new List<Tok>();
        if (data.length > 0) toks.add({p: data, s: true});
        return toks;
    }
    function makeDirect() {
        return new List<Tok>();
    }
    static function main() {
        new ForwardReturnInference("hi");
        trace("CONFORMANCE_OK");
    }
}
