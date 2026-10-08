typedef FromShape = {x:Int, ?y:Int, ?z:Int};
class ShapeConversionCount {public static var calls=0;}
@:forward abstract FromRecord(FromShape) from FromShape {
  @:from public static function convert(value:{x:Int,?y:Int}):FromRecord {
    ShapeConversionCount.calls++;
    return {x:1,y:value.y,z:0};
  }
}
class StructuralFromMethods {
  static function take(value:FromRecord):Int return value.x;
  static function main() {
    var matching:FromRecord={x:0,y:null};
    if (matching.x != 1 || ShapeConversionCount.calls != 1) throw "nullable structural conversion";
    var fewer:FromRecord={x:0};
    if (fewer.x != 0 || ShapeConversionCount.calls != 1) throw "fewer fields direct clause";
    var filled:FromRecord={x:0,y:2};
    if (filled.x != 1 || filled.y != 2 || ShapeConversionCount.calls != 2) throw "structural value conversion";
    var more:FromRecord={x:0,y:2,z:3};
    if (more.x != 0 || ShapeConversionCount.calls != 2) throw "more fields direct clause";
    var source:{x:Int,?y:Int}={x:0,y:4};
    var variable:FromRecord=source;
    if (variable.x != 1 || variable.y != 4 || ShapeConversionCount.calls != 3) throw "structural variable conversion";
    if (take({x:0,y:5}) != 1 || ShapeConversionCount.calls != 4) throw "structural argument conversion";
    Sys.println("CONFORMANCE_OK");
  }
}
