package support;
class SlotBase {
  public function new() {}
  public function value(argument:Int):Float return argument + 2.0;
  public function result(argument:Int):Float return argument + 2.0;
  public function inherited(argument:Float):Float return argument;
}
