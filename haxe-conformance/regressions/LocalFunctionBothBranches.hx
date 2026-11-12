// A local function called in both arms of a branch, one arm through a loop,
// is the same function value in the other arm.
class LocalFunctionBothBranches {
	static function tracks(v:String):String {
		function track(t:String):String {
			return t + "!";
		}
		return [
			for (part in v.split(" "))
				part == "repeat" ? [for (t in v.split(" ")) track(t)].join(" ") : track(part)
		].join(" ");
	}

	static function blocks(v:String):String {
		function track(t:String):String {
			return "<" + t + ">";
		}
		var out = [];
		for (part in v.split(" ")) {
			if (part == "all") {
				for (t in v.split(" "))
					out.push(track(t));
			} else
				out.push(track(part));
		}
		return out.join("");
	}

	static function main() {
		var got = tracks("1fr 12") + "|" + tracks("repeat x") + "|" + blocks("a all");
		if (got != "1fr! 12!|repeat! x! x!|<a><a><all>")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
