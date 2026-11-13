class Select {
	public static var seen:Array<String> = [];

	public static function register(o:Option, tag:String):Void {
		seen.push(o.label + ":" + tag);
	}
}
