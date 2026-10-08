abstract AbiCallable<T>(T->T) from T->T {
  public inline function apply(value:T):T return this(value);
}
class AbiBox {
  public var value:Int;
  public function new(value:Int) this.value = value;
}
class GenericFunctionAbi {
  static function call<T>(functionValue:T->T, value:T):T return functionValue(value);
  static function get<T>(functionValue:Void->T):T return functionValue();
  static function mixed<T>(functionValue:(Int,T)->T, value:T):T return functionValue(2,value);
  static function consume<T>(functionValue:T->Void, value:T):Void functionValue(value);
  static function defaulted<T>(functionValue:(T,?T)->T, value:T):T return functionValue(value);
  static function skipped<T>(functionValue:(?Int,String)->T):T return functionValue("value");
  static function main() {
    var f:AbiCallable<Float> = (value:Float)->value * 2;
    if (f.apply(1.5) != 3.0) throw "abstract Float call";
    if (call((value:Float)->value * 2,1.5) != 3.0) throw "generic Float call";
    if (call((value:Int)->value + 1,41) != 42) throw "generic Int call";
    if (call((value:String)->value + "!","value") != "value!") throw "generic String call";
    if (call((value:Bool)->!value,true) != false) throw "generic Bool call";
    if (get(()->1.5) != 1.5) throw "generic Float result";
    if (mixed((multiplier:Int,value:Float)->multiplier * value,1.5) != 3.0) throw "mixed generic call";
    var box = call((value:AbiBox)->new AbiBox(value.value + 1),new AbiBox(41));
    if (box.value != 42) throw "generic reference call";
    var consumed = 0.0;
    consume((value:Float)->{ consumed = value; },1.5);
    if (consumed != 1.5) throw "generic Void call";
    var optional = defaulted(function(value:Float, ?other:Float):Float return other == null ? value : other,1.5);
    if (optional != 1.5) throw "generic optional call";
    var shifted = skipped(function(?unused:Int,text:String):Float return text.length * 1.5);
    if (shifted != 7.5) throw "generic skipped optional call";
    Sys.println("CONFORMANCE_OK");
  }
}
