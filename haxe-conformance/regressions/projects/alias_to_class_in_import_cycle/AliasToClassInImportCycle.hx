// A typedef naming a class by its qualified path, declared in a module the
// class's own imports reach, is that class: its fields keep their types.
import kit.Rig;
import core.Frame;

class AliasToClassInImportCycle {
	static function main() {
		var rig = new Rig();
		var f = @:privateAccess new Frame({min: new math.V(-4), max: new math.V(4)});
		rig.prepare(f);
		if (rig.got != 36)
			throw rig.got;
		Sys.println("CONFORMANCE_OK");
	}
}
