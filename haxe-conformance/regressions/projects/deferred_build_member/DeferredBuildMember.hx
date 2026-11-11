// A member a deferred build adds is typed as declared where another module calls it.
import t.Size;
import t.Sizes;

class DeferredBuildMember {
	static function main() {
		var sizes = Sizes.defaults();
		var got = [sizes.get(Size.Small) * 2, sizes.get(Size.Large) + 0.5, Size.Large.of(sizes)].join(",");
		if (got != "8,16.5,16")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
