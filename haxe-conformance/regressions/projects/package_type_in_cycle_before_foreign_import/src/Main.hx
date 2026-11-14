class Main {
	static function main() {
		var got = p.ui.Form.check();
		if (got != "valid true form")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
