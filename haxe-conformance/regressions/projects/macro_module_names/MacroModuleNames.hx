class Base {
	public function new() {}
}

interface IA {}

class Child extends Base implements IA implements lib.Shader {
	public var x = 1;
	static var y = 2;
}

class MacroModuleNames {
	static function main() {
		var want = "iface=false super=Base ifaces=IA,Shader | alpha=Alpha twice=8 limit=7 | all=Alpha | fields=0 statics=0"
			+ " | name=Beta index=1 eq=true | tfun=TFun:2 tint=TInt";
		if (Child.REPORT != want)
			throw Child.REPORT + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
