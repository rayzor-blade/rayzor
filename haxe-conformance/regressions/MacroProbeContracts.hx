import haxe.macro.Context;
import haxe.macro.Expr;

private abstract SourceNumber(Int) to Int {
    public function new(value:Int) this = value;
    function hidden():Int return this;
}
private abstract TargetNumber(Int) from Int {}
private abstract BoxedNumber({value:Int}) {
    public function new(value:Int) this = {value:value};
    @:to public function toInt():Int return this.value;
}
private enum abstract Mode(Int) {
    var On = 1;
    static var ordinary = 2;
}
private class Parent {
    public function new() {}
    public function answer():Int return 42;
}
private class Child extends Parent { public function new() super(); }
private class NoConstructor { public var value = 1; }
private abstract class AbstractParent { public function new() {} }

class MacroProbeContracts extends Parent {
    static macro function rejects(value:Expr):Expr {
        var failed = false;
        try Context.typeof(value) catch (_:Dynamic) failed = true;
        return macro $v{failed};
    }
    extern static inline function inlineOnly(value:Int):Int return value + 1;

    public function new() super();
    function checkSuper() {
        if (rejects(super.answer())) throw "valid super call";
        if (!rejects(var value = super)) throw "super value";
    }
    static function main() {
        var source = new SourceNumber(1);
        if (rejects((source:Int))) throw "outgoing cast";
        if (rejects((1:TargetNumber))) throw "incoming cast";
        if (!rejects((source:TargetNumber))) throw "chained casts";
        if (!rejects(source.hidden())) throw "private abstract member";
        var callback:()->BoxedNumber = () -> new BoxedNumber(1);
        if (!rejects((callback:()->Int))) throw "function conversion";
        var numbers:()->Int = () -> 1;
        if (rejects((numbers:()->Float))) throw "numeric return";
        var children:Array<Child> = [new Child()];
        if (!rejects((children:Array<Parent>))) throw "array variance";
        if (rejects(var parents:Array<Parent> = [new Child()])) throw "contextual array";
        if (!rejects(new NoConstructor())) throw "missing constructor";
        if (!rejects(new AbstractParent())) throw "abstract class";
        if (!rejects(inlineOnly)) throw "extern inline closure";
        if (!rejects(ordinary)) throw "ordinary enum abstract static";
        if (!rejects(try {} catch (e:Parent) {} catch (e:Child) {})) throw "covered catch";
        if (rejects(try {} catch (e:Child) {} catch (e:Parent) {})) throw "ordered catches";
        new MacroProbeContracts().checkSuper();
        Sys.println("CONFORMANCE_OK");
    }
}
