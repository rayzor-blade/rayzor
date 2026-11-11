package comp;

import lay.Into;

class Library {
	// Refers back to Badge, so Badge's module is lowered first.
	public static function defaultBadge():Badge
		return new Badge("none");

	public static function part(name:String, data:Map<String, Into<String>>):String {
		var out = [];
		for (key => value in data)
			out.push(key + "=" + switch (value : Kind<String>) {
				case Const(x): x;
				case Missing: "missing";
			});
		out.sort(Reflect.compare);
		return name + ":" + out.join(";");
	}
}
