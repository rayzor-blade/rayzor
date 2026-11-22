import lib.Value;
import lib.Writer;

class StaticGenericCallOtherModule {
	static function main() {
		var w = new Writer();
		w.apply(1, 5);
		w.apply(2, Bound(() -> "x"));
		w.apply(3, 2.5);
		w.unset(4);
		w.unset(5);
		var got = w.log.join(" ");
		if (got != "1=null>1=5 2=null>2=x 3=null>3=2.5 4=1>4=7")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
