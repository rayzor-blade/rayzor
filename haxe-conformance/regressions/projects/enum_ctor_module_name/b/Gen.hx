package b;

enum RG<T> {
	Cst(v:T);
	Sig(s:Array<T>);
}

abstract IR<T>(RG<T>) from RG<T> to RG<T> {
	@:from public static function ofConst<T>(v:T):IR<T>
		return Cst(v);
}

class GenFns {
	public static function take(r:IR<Float>):String
		return switch (r : RG<Float>) {
			case Cst(v): "" + v;
			case Sig(s): "sig" + s.length;
		}
}
