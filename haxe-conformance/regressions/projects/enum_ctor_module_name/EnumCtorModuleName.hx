import b.Gen;

class EnumCtorModuleName {
	static function main() {
		var implicit = GenFns.take(5.5);
		var direct = GenFns.take(Cst(2.5));
		var viaStatic = GenFns.take(IR.ofConst(1.5));
		if (implicit != "5.5" || direct != "2.5" || viaStatic != "1.5")
			throw '$implicit $direct $viaStatic';
		Sys.println("CONFORMANCE_OK");
	}
}
