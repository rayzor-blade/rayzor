class TypeTokensInDynamicArrays {
	static var types:Array<Dynamic> = [null, Int, String, Bool, Float, Array, Date];
	static function same<T>(a:T, b:T):Bool return a == b;
	static function sameArray<T>(a:T, b:T):Bool return a == b;

	static function main() {
		if (types.length != 7) throw "type values omitted";
		if (Type.getClassName(types[2]) != "String") throw "boxed String class";
		if (Type.getClassName(types[5]) != "Array") throw "boxed Array class";
		if (Type.getClassName(Type.resolveClass("String")) != "String") throw "resolve String";
		if (!same(Type.resolveClass("String"), types[2])) throw "boxed class equality";
		if (!sameArray(Type.resolveClass("Array"), types[5])) throw "boxed array class equality";
		Sys.println("CONFORMANCE_OK");
	}
}
