import b.Store;
import b.Store.Mixed;

class SingleCrossModule {
	static function check(label:String, got:Float, want:Float):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var s = new Store(1);
		check("constructor", s.a, 1);
		s.setA(120);
		check("method Int literal", s.a, 120);
		s.setF(1.5);
		check("method Float field", s.f, 1.5);
		Store.setLast(1.5);
		check("static Float literal", Store.last, 1.5);
		var m = new Mixed(7, 8, 9);
		check("mixed i", m.i, 7);
		check("mixed s", m.s, 8);
		check("mixed f", m.f, 9);
		trace("CONFORMANCE_OK");
	}
}
