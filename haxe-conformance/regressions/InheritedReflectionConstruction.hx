class InheritedCallbackParent {
  public static var constructions:Int=0;
  public var callback:()->Bool = function() return true;
  public var value:Int;
  function new(value:Int=17) {
    constructions++;
    this.value=value;
  }
}
class InheritedCallbackChild extends InheritedCallbackParent {}
class InheritedReflectionConstruction {
  static function main() {
    var child=Type.createInstance(InheritedCallbackChild,[23]);
    if (InheritedCallbackParent.constructions != 1) throw "inherited constructor";
    if (child.value != 23) throw "inherited constructor arguments";
    if (!child.callback()) throw "inherited callback initializer";
    if (Type.getClass(child) != InheritedCallbackChild) throw "derived identity";
    var forward=Type.createInstance(ForwardCallbackChild,[]);
    if (ForwardCallbackParent.constructions != 1) throw "forward inherited constructor";
    if (!forward.callback()) throw "forward callback initializer";
    if (Type.getClass(forward) != ForwardCallbackChild) throw "forward derived identity";
    Sys.println("CONFORMANCE_OK");
  }
}
private class ForwardCallbackParent {
  public static var constructions:Int=0;
  public var callback = function() return true;
  function new() constructions++;
}
private class ForwardCallbackChild extends ForwardCallbackParent {}
