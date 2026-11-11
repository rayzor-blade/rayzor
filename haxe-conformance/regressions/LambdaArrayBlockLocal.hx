// A local declared in a block inside an array literal belongs to the enclosing lambda.
class Box {
	public var v = 0;
	public var kids:Array<Box>;

	public function new(?kids:Array<Box>)
		this.kids = kids == null ? [] : kids;
}

class LambdaArrayBlockLocal {
	static function main() {
		var inArray = () -> [
			{
				var c = new Box();
				c.v = 7;
				c;
			}
		];
		var inCtor = () -> new Box([
			{
				var d = new Box();
				d.v = 8;
				d;
			}
		]);
		var inObject = () -> {
			box: {
				var e = new Box();
				e.v = 9;
				e;
			}
		};
		var got = [inArray()[0].v, inCtor().kids[0].v, inObject().box.v].join(",");
		if (got != "7,8,9")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
