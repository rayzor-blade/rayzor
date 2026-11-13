// A conditional of a Null<Single> field and a literal, typed as Single and
// stored in a Single field, is a Single on either arm.
#if rayzor
typedef TextNumber = Single;
#else
typedef TextNumber = Float;
#end

typedef TextAttrs = {?color:Int, ?fontSize:TextNumber, ?wrap:Bool};

class TextNode {
	public var fs:TextNumber;
	public var wrap:Bool;

	public function new(?attr:TextAttrs) {
		var size:TextNumber = attr != null && attr.fontSize != null ? attr.fontSize : 16.0;
		var w = attr != null && attr.wrap != null ? attr.wrap : true;
		fs = size;
		wrap = w;
	}
}

class NullableSingleConditionalToSingle {
	static function main() {
		var nodes = [new TextNode(), new TextNode({}), new TextNode({fontSize: 20}), new TextNode({wrap: false}), new TextNode({color: 5})];
		var got = [for (n in nodes) n.fs + "/" + n.wrap].join(",");
		if (got != "16/true,16/true,20/true,16/false,16/true")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
