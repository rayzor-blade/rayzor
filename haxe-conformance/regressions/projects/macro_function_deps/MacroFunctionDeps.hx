class MacroFunctionDeps {
	static function main() {
		var got = [DepsMac.onCtor(), DepsMac.onVar(), "" + DepsMac.onField().a, MacUser.onCtor(), MacUser.onVar(), MacUser.withArg()].join(",");
		if (got != "ctor,var,1,circle 3,circle 4,square x2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
