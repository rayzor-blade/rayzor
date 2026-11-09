package rayzor.runtime;

/**
	The table `untyped __resources__()` returns. The compiler calls `add` for
	each resource at the start of `haxe.Resource.__init__`, after every macro
	has run, so the table holds the resources macros added late.
**/
@:keep
class Resources {
	static var entries:Array<{name:String, data:String, str:String}>;

	public static function add(name:String, data:String):Void {
		if (entries == null)
			entries = [];
		entries.push({name: name, data: data, str: null});
	}

	public static function table():Array<{name:String, data:String, str:String}> {
		return entries == null ? [] : entries;
	}
}
