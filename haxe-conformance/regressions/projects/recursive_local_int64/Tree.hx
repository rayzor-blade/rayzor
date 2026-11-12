import haxe.Int64;

class Tree {
	public static function children(id:Int64):Array<Int64> {
		return [for (i in 0...2) Int64.make(1, id.low + i + 1)];
	}
}
