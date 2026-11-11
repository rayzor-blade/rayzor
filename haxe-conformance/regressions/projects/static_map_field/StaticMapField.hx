// A static field typed Map<K,V> holds the map its initializer builds.
class StaticMapField {
	static function main() {
		var got = Names.report();
		if (got != "true,1,true,seven,v,a1+div2")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
