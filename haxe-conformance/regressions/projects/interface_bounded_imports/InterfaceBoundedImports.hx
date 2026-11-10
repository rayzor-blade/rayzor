import shapes.Shapes;

class InterfaceBoundedImports {
	static function idRP(r:RP):Int
		return switch r {
			case BP(s): s.id();
		};

	static function ptrRS(r:RS):Int
		return switch r {
			case BS(s): s.ptr;
		};

	static function getRS(r:RS):Int
		return switch r {
			case BS(s): s.get();
		};

	static function take(h:Hd):Hd
		return h;

	static function handleOf<T:IH>(v:T):Hd
		return take(v.h);

	static function check(label:String, got:Int, want:Int):Void {
		if (got != want)
			throw label + ": " + got + " != " + want;
	}

	static function main() {
		var f = new SBField<Col>(new Col(65280));
		check("imported bounded field", f.current.rgb, 65280);
		var c:Col = f.current;
		check("imported bounded field into a local", c.ptr, 65281);
		var sv = new SV<Col>(new Col(10));
		check("imported bounded get", sv.get().rgb, 10);
		check("imported bounded method", sv.ptrOf(), 11);

		check("class into an interface payload", idRP(BP(new PlainImpl())), 77);
		var i:IPlain = new PlainImpl();
		check("interface value into a payload", idRP(BP(i)), 77);
		var r = BP(new PlainImpl());
		check("local enum of a class payload", switch r {
			case BP(s): s.id();
		}, 77);

		var s:Sg<Int> = new SgI(2);
		check("abstract payload property", ptrRS(BS(s)), 77);
		check("abstract payload from a class", ptrRS(BS(new SgI(2))), 77);
		check("abstract payload method", getRS(BS(s)), 2);
		var hc = new HCol(31);
		check("imported interface field passed on", handleOf(hc) == hc.h ? 1 : 0, 1);
		check("imported interface field read", handleOf(hc).id, 31);
		Sys.println("CONFORMANCE_OK");
	}
}
