// An optional `Single` field, through an alias or not, is nullable: a given
// zero is a value, an omitted field is null.
typedef OptionalSingleS32 = Single;

typedef OptionalSingleAttr = {?width:OptionalSingleS32, ?height:Single};

class OptionalSingleField {
	static function width(attr:Null<OptionalSingleAttr>, fallback:Int):Single
		return attr != null && attr.width != null ? attr.width : (fallback : Single);

	static function main() {
		var got = width({width: 50}, 24) + " " + width({width: 0}, 24) + " " + width({}, 24) + " " + width(null, 24) + " "
			+ ({height: 3} : OptionalSingleAttr).height;
		if (got != "50 0 24 24 3")
			throw got;
		Sys.println("CONFORMANCE_OK");
	}
}
