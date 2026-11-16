import lib.Mod;

class GenericTemplateCallOtherModule {
	static function main() {
		var n = new Node();
		Node.write(n, (66 : Prop<Shadow>), new Shadow(1));
		Node.write(n, (4 : Prop<Float>), 2.5);
		Node.write(n, (5 : Prop<Int>), 7);
		Node.write(n, (66 : Prop<Shadow>), new Shadow(2));
		for (w in n.batch.writes)
			w();
		if (n.seen != "66:S1:true 4:2.5:true 5:7:true 66:S1:false ")
			throw n.seen;
		Sys.println("CONFORMANCE_OK");
	}
}
