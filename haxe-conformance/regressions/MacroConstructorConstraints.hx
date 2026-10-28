import haxe.macro.Context;
import haxe.macro.Expr;

private class Parent { public function new() {} }
private class Child extends Parent { public function new() super(); }
private class ParentFactory { public function new(value:Parent) {} }
private class ChildFactory { public function new(value:Child) {} }

class MacroConstructorConstraints {
    static macro function rejects(value:Expr):Expr {
        var failed = false;
        try Context.typeof(value) catch (_:Dynamic) failed = true;
        return macro $v{failed};
    }
    @:generic static function create<A, C:haxe.Constraints.Constructible<A->Void>>(value:A, type:Class<C>)
        return new C(value);

    static function main() {
        var parent = new Parent();
        var child = new Child();
        if (rejects(create(parent, ParentFactory))) throw "matching constructor";
        if (rejects(create(child, ChildFactory))) throw "matching child constructor";
        if (rejects(create(child, ParentFactory))) throw "contravariant constructor";
        if (!rejects(create(parent, ChildFactory))) throw "narrower constructor";
        Sys.println("CONFORMANCE_OK");
    }
}
