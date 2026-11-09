class Plain {
	public var x:Int = 3;

	public function new() {}
}

typedef PlainAlias = Plain;

class TypedefClassValue {
	static function check(value:Bool, label:String):Void {
		if (!value)
			throw label;
	}

	static function main():Void {
		var alias:Dynamic = PlainAlias;
		check(alias != null, "typedef read as a value");
		check(alias == Plain, "typedef value is its class");
		var k:Class<Dynamic> = Type.getClass(new Plain());
		var matched = switch (k) {
			case cast PlainAlias: "plain";
			default: "other";
		}
		check(matched == "plain", "typedef as a class pattern");
		var bytesClass:Dynamic = haxe.io.Bytes;
		check(bytesClass != null, "std typedef read as a value");
		check(haxe.Serializer.run(new Plain()) == "cy5:Plainy1:xi3g", "serialize a class instance");
		trace("CONFORMANCE_OK");
	}
}
