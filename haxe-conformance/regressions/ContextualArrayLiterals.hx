private class ContextCell {
 public var value:Int;
 public function new(value:Int) this.value = value;
 public static function make(value:Int):ContextCell return new ContextCell(value);
}
class ContextualArrayLiterals {
 static var mixed:Array<Dynamic> = [0, "hello", " world", 0.5];
 var cells:Array<ContextCell>;
 var numbers:Array<Float>;
 public function new() {
  cells = [ContextCell.make(7), ContextCell.make(8)];
  var value:Dynamic = new ContextCell(9);
  cells = [value];
  if (cells[0].value != 9) throw "boxed field initializer";
  var n:Dynamic = 1.5;
  numbers = [n];
  if (numbers[0] != 1.5) throw "boxed float field initializer";
 }
 static function main() {
  var instance = new ContextualArrayLiterals();
  var value:Dynamic = new ContextCell(11);
  var cells:Array<ContextCell> = [value];
  if (cells[0].value != 11) throw "boxed local initializer";
  var n:Dynamic = 1.5;
  var numbers:Array<Float> = [n];
  if (numbers[0] != 1.5) throw "boxed float local initializer";
  var promoted:Array<Float> = [1, 2];
  if (promoted[0] != 1.0 || promoted[1] != 2.0) throw "literal numeric promotion";
  if (mixed[1] + mixed[2] != "hello world") throw "inferred mixed initializer";
  var nullable:Array<Null<Float>> = [null, 1.5];
  if (nullable[0] != null || nullable[1] != 1.5) throw "nullable float slots";
  var nullableBools:Array<Null<Bool>> = [null, true, false];
  if (nullableBools[0] != null || nullableBools[1] != true || nullableBools[2] != false) throw "nullable bool slots";
  Sys.println("CONFORMANCE_OK");
 }
}
