// A generic class is compiled once, so `first(): Null<T>` cannot know T and
// boxes whatever bits it stored as an Int. The CALLER knows T, and for a
// reference type the stored bits are the pointer -- they must be read back out
// of the box rather than the box itself being used as the value. Before this
// `first().p` on a List<{p:String}> read the box's type tag (3) as the field.
private typedef Tok = { p:String, s:Bool };
private class Node { public var v:Int; public function new(v:Int) this.v = v; }

class ErasedGenericReturn {
    // the receiver is a PARAMETER, so its type carries the argument
    static function anonFirst(l:List<Tok>):String return l.first().p;
    static function anonPop(l:List<Tok>):String return l.pop().p;
    static function classFirst(l:List<Node>):Int return l.first().v;
    static function intFirst(l:List<Int>):Int return l.first();
    static function main() {
        var lt = new List<Tok>();
        lt.add({p:"pp", s:true}); lt.add({p:"qq", s:false});
        if (anonFirst(lt) != "pp") throw "anon first";
        if (anonPop(lt) != "pp") throw "anon pop";
        if (anonPop(lt) != "qq") throw "anon pop, second";
        var ln = new List<Node>();
        ln.add(new Node(7));
        if (classFirst(ln) != 7) throw "class first";
        var li = new List<Int>();
        li.add(42);
        if (intFirst(li) != 42) throw "int first";
        trace("CONFORMANCE_OK");
    }
}
