package b;

class M {
	public static function four(a:Single, p:Int, q:Int, r:Int):Single
		return a + p + q + r;

	public static function five(a:Single, p:Int, q:Int, r:Int, s:Int):Single
		return a + p + q + r + s;

	public static function seven(a:Single, b:Single, p:Int, q:Int, r:Int, s:Int, t:Int):Single
		return a + b + p + q + r + s + t;

	public static function nine(a:Single, b:Single, c:Single, d:Single, e:Single, f:Single, g:Single, h:Single, i:Single):Single
		return a + b + c + d + e + f + g + h + i;

	public function new() {}

	public function sized(name:String, size:Single, line:Single, w:Int, h:Int, x:Int):Single
		return size + line + w + h + x;
}
