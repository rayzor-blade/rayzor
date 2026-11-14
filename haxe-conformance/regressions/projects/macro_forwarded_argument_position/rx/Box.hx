package rx;

class Box {
	public static var made = 0;

	public final value:String;
	public final kind:String;

	public function new(value:String, kind:String) {
		this.value = value;
		this.kind = kind;
		made++;
	}
}
