package lib;

// An interface whose method returns a typedef declared beside it, and one
// whose result typedef is another module's.
typedef Token = {final from:Int; final to:Int; final kind:String;}
typedef Highlight = {final tokens:Array<Token>; final state:String;}

interface Highlighter {
	public function highlight(line:String, state:String):Highlight;
}

interface Measurer {
	public function measure(line:String):lib.Shapes.Extent;
}
