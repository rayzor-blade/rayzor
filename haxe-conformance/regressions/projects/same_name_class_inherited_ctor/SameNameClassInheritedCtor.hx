// `new lib.ui.Label(...)` inside another class named Label constructs
// lib.ui.Label through the constructor it inherits from a generic base.
class SameNameClassInheritedCtor {
	static function main() {
		var l = new lib.components.Label({required: true});
		if (l.out != "components.Label>ui.Label")
			throw l.out;
		Sys.println("CONFORMANCE_OK");
	}
}
