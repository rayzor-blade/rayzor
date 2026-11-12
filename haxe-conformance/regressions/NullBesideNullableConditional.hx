class NullBesideNullableCtx {
	public final props:Null<Map<String, String>>;

	public function new(props:Null<Map<String, String>>)
		this.props = props;
}

// `cond ? null : map.get(k)` is Null<String>: the value stays a String through a
// try that returns it.
class NullBesideNullableConditional {
	static function validate(name:String, value:String):Void {
		switch name {
			case "glass-blur":
				var n = Std.parseFloat(value);
				if (!Math.isFinite(n) || n < 0)
					throw "expected a nonnegative length";
			case "glass-mode":
				if (["liquid", "frosted"].indexOf(StringTools.trim(value).toLowerCase()) < 0)
					throw "expected liquid or frosted";
		}
	}

	static function glass(c:NullBesideNullableCtx):String {
		function setting(name:String, fallback:String):String {
			var value = c.props == null ? null : c.props.get(name);
			if (value == null)
				return fallback;
			return try {
				validate(name, value);
				value;
			} catch (_:String)
				fallback;
		}
		var tint = setting("glass-tint", "rgba(255,255,255,0.1)");
		var blur = setting("glass-blur", "12px");
		var mode = setting("glass-mode", "liquid");
		return tint + " | " + blur + " | " + mode;
	}

	static function main() {
		var got = glass(new NullBesideNullableCtx(["glass-blur" => "2px"])) + "|" + glass(new NullBesideNullableCtx(null));
		if (got != "rgba(255,255,255,0.1) | 2px | liquid|rgba(255,255,255,0.1) | 12px | liquid")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
