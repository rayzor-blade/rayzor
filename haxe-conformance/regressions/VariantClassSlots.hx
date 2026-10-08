class VariantSlotBase {
  public function new() {}
  public function widened(value:Int):Float return value + 2.0;
  public function narrowed(value:Int):Float return value + 2.0;
  public function integer(value:Int):Int return value;
  public function inherited(value:Float):Float return value;
  public function mixed(value:Int, text:String, enabled:Bool):Float return enabled ? value + text.length : 0.0;
}
class VariantSlotChild extends VariantSlotBase {
  override public function widened(value:Float):Float return value + 1.0;
  override public function narrowed(value:Float):Int return Std.int(value) + 1;
  override public function integer(value):Int return value;
  override public function inherited(renamed):Float return renamed + 1.0;
  override public function mixed(value:Float, text, enabled):Int return enabled ? Std.int(value) + text.length : 0;
  public function parentCall(value:Int):Float return super.widened(value);
}
class VariantSlotLeaf extends VariantSlotChild {
  override public function widened(value:Float):Float return value + 3.0;
}
class VariantClassSlots {
  static function check(actual:Float, expected:Float):Void {
    if (actual != expected) throw "numeric slot value";
  }
  static function main() {
    var base = new VariantSlotBase();
    var child = new VariantSlotChild();
    var parent:VariantSlotBase = child;
    check(base.widened(40), 42.0);
    check(parent.widened(41), 42.0);
    check(child.widened(41.5), 42.5);
    check(parent.narrowed(41), 42.0);
    check(child.narrowed(41.5), 42.0);
    check(parent.widened(-42), -41.0);
    check(child.parentCall(40), 42.0);
    check(parent.inherited(41.5), 42.5);
    check(child.inherited(41.5), 42.5);
    if (parent.integer(2147483647) != 2147483647) throw "positive Int limit";
    if (parent.integer(-2147483647 - 1) != -2147483647 - 1) throw "negative Int limit";
    check(parent.mixed(39, "abc", true), 42.0);
    check(child.mixed(39.5, "abc", true), 42.0);
    check(parent.mixed(39, "abc", false), 0.0);
    var leaf = new VariantSlotLeaf();
    var asParent:VariantSlotBase = leaf;
    var asChild:VariantSlotChild = leaf;
    check(asParent.widened(39), 42.0);
    check(asChild.widened(39.5), 42.5);
    Sys.println("CONFORMANCE_OK");
  }
}
