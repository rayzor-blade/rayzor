package support;
class SlotChild extends SlotBase {
  override public function value(argument:Float):Float return argument + 1.0;
  override public function result(argument:Float):Int return Std.int(argument) + 1;
  override public function inherited(renamed):Float return renamed + 1.0;
}
