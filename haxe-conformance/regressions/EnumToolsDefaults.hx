enum Shade {
	Light;
	Dark;
}

enum Kind {
	Alpha;
	Beta(n:Int);
}

class EnumToolsDefaults {
	static function check(label:String, got:String, want:String):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		check("createAll", Shade.createAll().join(","), "Light,Dark");
		check("createByName", Std.string(Shade.createByName("Dark")), "Dark");
		check("createByIndex", Std.string(Shade.createByIndex(0)), "Light");
		check("getConstructors", Shade.getConstructors().join(","), "Light,Dark");
		check("Type.createEnum", Std.string(Type.createEnum(Shade, "Dark")), "Dark");
		check("Type.createEnumIndex", Std.string(Type.createEnumIndex(Shade, 0)), "Light");
		var made:Shade = Shade.createByName("Light");
		check("typed result", made == Light ? "yes" : "no", "yes");
		var b = Beta(5);
		check("equals", Std.string(b.equals(Beta(5))), "true");
		check("not equals", Std.string(b.equals(Beta(6))), "false");
		check("getName", b.getName(), "Beta");
		check("getIndex", Std.string(b.getIndex()), "1");
		check("getParameters", Std.string(b.getParameters()), "[5]");
		var s = Dark;
		check("tag getName", s.getName(), "Dark");
		check("tag getIndex", Std.string(s.getIndex()), "1");
		check("tag getParameters", Std.string(s.getParameters()), "[]");
		check("tag equals", Std.string(s.equals(Dark)), "true");
		Sys.println("CONFORMANCE_OK");
	}
}
