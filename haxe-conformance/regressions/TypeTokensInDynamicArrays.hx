class TypeTokensInDynamicArrays {
	static var types:Array<Dynamic> = [null, Int, String, Bool, Float, Array, Date];

	static function main() {
		if (types.length != 7) throw "type values omitted";
		if (Type.getClassName(types[2]) != "String") throw "boxed String class";
		if (Type.getClassName(types[5]) != "Array") throw "boxed Array class";
		if (Type.getClassName(Type.resolveClass("String")) != "String") throw "resolve String";
		Sys.println("CONFORMANCE_OK");
	}
}
