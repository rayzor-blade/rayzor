// An unqualified type name in package p is p's type before a root-package namesake.
class PackageShadow {
	static function main() {
		var got = [p.User.describe(), p.User.make().tag, new Thing().tag].join(",");
		if (got != "p 11,p,root")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
