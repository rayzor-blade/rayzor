// A Dynamic object's field is read from the object, even when a runtime
// class has a parameterless method of that name.
class DynamicFieldNamedLikeRuntimeMethod {
	static function main() {
		var d:Dynamic = {toUpperCase: 3, hasNext: 4, isEmpty: 6, pop: 7, keys: 8, width: 9};
		var got = d.toUpperCase + " " + d.hasNext + " " + d.isEmpty + " " + d.pop + " " + d.keys + " " + d.width;
		if (got != "3 4 6 7 8 9")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
