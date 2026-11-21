package lib;

import lib.RadioGroup;
import lib.Component;
import lib.Element;

class Form {
	public static var seen = 0;
	public static var last = "";

	public static function make():RadioGroup
		return new RadioGroup({value: "f"});
}
