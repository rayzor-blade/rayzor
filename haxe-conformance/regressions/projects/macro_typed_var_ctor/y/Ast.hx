package y;

typedef Fn = {name:String, n:Int};

enum Def {
	EBlock(items:Array<Int>);
	EFunction(f:Fn);
	EBreak;
}
