interface SlotFloatView {
  public function get():Float;
  public function widened(value:Int):Float;
  public function self():SlotFloatView;
  public function absent():SlotFloatView;
  public function consume(value:SlotObject):Void;
  public function echo(value:SlotObject):SlotFloatView;
}
interface SlotIntView {
  public function get():Int;
}
class SlotObject implements SlotFloatView implements SlotIntView {
  public function new() {}
  public function get():Int return 42;
  public function widened(value:Float):Int return Std.int(value) + 1;
  public function self():SlotObject return this;
  public function absent():SlotObject return null;
  public function consume(value:SlotFloatView):Void {
    if (value.get() != 42.0) throw "class argument to interface implementation";
  }
  public function echo(value:SlotFloatView):SlotFloatView return value;
}
class SlotChild extends SlotObject {}
class InterfaceSlotVariants {
  static function same<T>(a:T, b:T):Bool return a == b;
  static function dynamicSame<T>(a:T, b:Dynamic):Bool return a == b;
  static function boxed<T>(value:T):Dynamic return value;
  static function text<T>(value:T):String return Std.string(value);
  static function main() {
    var object = new SlotObject();
    var floats:SlotFloatView = object;
    var integers:SlotIntView = object;
    if (floats.get() != 42.0) throw "Float interface result";
    if (integers.get() != 42) throw "Int interface result";
    if (floats.widened(41) != 42.0) throw "numeric interface parameter and result";
    if (object.widened(41.5) != 42) throw "native class parameter";
    var returned:SlotFloatView = floats.self();
    if (returned.get() != 42.0) throw "class result to interface slot";
    if (returned != floats) throw "returned interface identity";
    if (!same(floats, returned)) throw "generic interface identity";
    if (!same(floats, object)) throw "generic interface and class identity";
    if (same(floats, new SlotObject())) throw "distinct interface objects";
    if (same(floats, null)) throw "interface against null";
    if (text(floats) != Std.string(object)) throw "generic interface rendering";
    var dynamicObject:Dynamic = object;
    var dynamicInterface:Dynamic = floats;
    if (dynamicObject != dynamicInterface) throw "Dynamic interface identity";
    if (!dynamicSame(floats, dynamicObject)) throw "generic interface against Dynamic";
    if (boxed(floats) != dynamicObject) throw "generic interface boxing";
    if (Std.downcast(floats, SlotObject) != object) throw "interface downcast";
    if (Std.downcast(object, SlotObject) != object) throw "class downcast";
    if (Std.downcast(floats, SlotChild) != null) throw "failed interface downcast";
    var empty:SlotFloatView = null;
    if (Std.downcast(empty, SlotObject) != null) throw "null interface downcast";
    var dynamicNull:Dynamic = null;
    if (!dynamicSame(empty, dynamicNull)) throw "generic null interface against Dynamic";
    if (floats.absent() != null) throw "null interface result";
    floats.consume(object);
    var echoed = floats.echo(object);
    if (echoed.get() != 42.0) throw "escaped interface argument";
    Sys.println("CONFORMANCE_OK");
  }
}
