enum Placement {
	Centered;
	Below(w:Int);
	Edge(side:String);
}

typedef Props = {?placement:Placement}

// `cond ? nullableEnum : FirstConstructor` passes an enum value, not a box,
// to a plain enum parameter; a Dynamic result still boxes it.
class NullEnumTernaryFirstConstructor {
	static function name(p:Placement):String
		return switch p {
			case Centered: "centered";
			case Below(w): "below " + w;
			case Edge(s): "edge " + s;
		}

	static function main() {
		var none:Props = {};
		var below:Props = {placement: Below(4)};
		var centered:Props = {placement: Centered};
		var flag = none.placement == null;
		var d:Dynamic = flag ? Centered : below.placement;
		var got = [
			name(none.placement != null ? none.placement : Centered),
			name(below.placement != null ? below.placement : Centered),
			name(centered.placement != null ? centered.placement : Below(1)),
			Std.string(d)
		].join(",");
		if (got != "centered,below 4,centered,Centered")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
