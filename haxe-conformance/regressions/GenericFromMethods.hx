class FromMethodCounter { public static var calls:Int=0; }
abstract ConvertedFunction<T>(T->T) {
  public inline function apply(value:T):T return this(value);
  @:from public static function convert<T>(value:T->T):ConvertedFunction<T> {
    FromMethodCounter.calls++;
    return cast value;
  }
}
abstract DirectFunction(String->Int) from String->Int {
  public inline function apply(value:String):Int return this(value);
  @:from public static function convert<T>(value:T->T):DirectFunction {
    throw "incompatible generic source selected";
  }
}
class GenericFromMethods {
  static function accept(value:ConvertedFunction<String>) {}
  static function main() {
    var converted:ConvertedFunction<String> = (value:String) -> value;
    if (FromMethodCounter.calls != 1) throw "generic function initializer conversion";
    if (converted.apply("value") != "value") throw "converted function value";
    accept((value:String) -> value);
    if (FromMethodCounter.calls != 2) throw "generic function argument conversion";
    var numeric:ConvertedFunction<Int> = (value:Int) -> value + 1;
    if (FromMethodCounter.calls != 3 || numeric.apply(41) != 42) throw "integer function conversion";
    var fractional:ConvertedFunction<Float> = (value:Float) -> value * 2;
    var floating:Float->Float = cast fractional;
    if (FromMethodCounter.calls != 4 || floating(1.5) != 3.0) throw "float function conversion";
    var direct:DirectFunction = (value:String) -> value.length;
    if (direct.apply("value") != 5) throw "direct function conversion";
    Sys.println("CONFORMANCE_OK");
  }
}
