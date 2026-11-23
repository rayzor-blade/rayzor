// An Array<Null<E>> of a tag-only enum hands its callbacks each element as a
// Null<E>, where the first constructor is a value, and a plain E taken as a
// Null<E> stays a value.
enum Phase {
	Closed;
	Opening;
	Waiting;
}

class TagEnumNullableArrayCallbacks {
	static function isNull(p:Null<Phase>):Bool
		return p == null;

	static function main() {
		final out = [];
		final arr:Array<Null<Phase>> = [Closed, null, Waiting];
		final opening:Null<Phase> = Opening;
		arr.push(opening);
		arr.push(null);
		out.push(arr.map(x -> x == null).join(","));
		out.push(arr.map(x -> Std.string(x)).join(","));
		out.push(Std.string(arr.filter(x -> x != null).length));
		final same = arr.map(x -> x);
		out.push(same.map(x -> x == Closed).join(","));
		var seen = [];
		for (x in arr)
			seen.push(x == null);
		out.push(seen.join(","));
		final plain:Phase = Closed;
		out.push((plain : Null<Phase>) == null ? "null" : "value");
		out.push(Std.string(isNull(plain)));
		final ok:Array<Phase> = [Closed, Waiting];
		out.push(ok.map(x -> (x : Null<Phase>) == null).join(","));
		out.push(ok.map(x -> isNull(x)).join(","));
		final got = out.join(" | ");
		final want = "false,true,false,false,true | Closed,null,Waiting,Opening,null | 3"
			+ " | true,false,false,false,false | false,true,false,false,true | value | false"
			+ " | false,false | false,false";
		if (got != want)
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
