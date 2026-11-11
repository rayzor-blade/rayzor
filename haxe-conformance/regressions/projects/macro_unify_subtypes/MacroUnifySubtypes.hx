// Context.unify accepts a class, or Null of one, for its superclasses and interfaces.
import Kinds;

class MacroUnifySubtypes {
	static function main() {
		var base = new Base();
		var sub = new Sub();
		var vc = new ViaChild();
		var un = new Unrelated();
		var got = [
			UnifyMac.u(sub, "Base"), UnifyMac.u(base, "Sub"), UnifyMac.u(base, "IBase"), UnifyMac.u(sub, "IBase"),
			UnifyMac.u(vc, "IBase"), UnifyMac.u(vc, "IChild"), UnifyMac.u(un, "IBase"), UnifyMac.u((null : Null<Base>), "IBase")
		].join(",");
		if (got != "true,false,true,true,true,true,false,true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
