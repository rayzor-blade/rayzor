class ErasedVectorValues {
 static function main() {
  var ints = new haxe.ds.Vector<Int>(3); ints[0]=65;ints[1]=66;ints[2]=81;
  var i:Dynamic=ints;
  var a:Int=untyped i[0]; var b:Int=untyped i[1]; var c:Int=untyped i[2];
  if (a != 65 || b != 66 || c != 81) throw "erased integer vector reads";
  untyped i[1]=17; if (ints[1] != 17) throw "erased integer vector write";
  var floats=new haxe.ds.Vector<Float>(2); floats[0]=1.5;floats[1]=-2.5;
  var f:Dynamic=floats; var first:Float=untyped f[0];
  if (first != 1.5) throw "erased float vector read";
  untyped f[1]=4.25; if (floats[1] != 4.25) throw "erased float vector write";
  var copied = haxe.ds.Vector.fromArrayCopy([1.25, 2.75]);
  var erasedCopy:Dynamic = copied;
  var copiedValue:Float = untyped erasedCopy[1];
  if (copiedValue != 2.75) throw "erased copied vector read";
  var cloned:Dynamic = copied.copy();
  var clonedValue:Float = untyped cloned[0];
  if (clonedValue != 1.25) throw "erased cloned vector read";
  var bools=new haxe.ds.Vector<Bool>(2);bools[0]=true;bools[1]=false;
  var t:Dynamic=bools;var truth:Bool=untyped t[0];
  if (!truth) throw "erased bool vector read";
  untyped t[1]=true; if (!bools[1]) throw "erased bool vector write";
  var words=new haxe.ds.Vector<haxe.Int64>(1);var large=haxe.Int64.make(1,-1);words[0]=large;
  var w:Dynamic=words;var word:haxe.Int64=untyped w[0];
  if (word != large) throw "erased Int64 vector read";
  var text=new haxe.ds.Vector<String>(2);text[0]="first";text[1]="second";
  var s:Dynamic=text;var value:String=untyped s[1];
  if (value != "second") throw "erased string vector read";
  untyped s[0]="new"; if (text[0] != "new") throw "erased string vector write";
  Sys.println("CONFORMANCE_OK");
 }
}
