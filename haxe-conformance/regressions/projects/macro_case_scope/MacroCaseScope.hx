@:build(CaseScopeMac.build())
class CaseScopeHolder {}

class MacroCaseScope {
	static function main() {
		var want = "Vec4,Outer,dotcrossfloat,TFloat,c";
		if (CaseScopeHolder.TEXT != want)
			throw CaseScopeHolder.TEXT + " != " + want;
		Sys.println("CONFORMANCE_OK");
	}
}
