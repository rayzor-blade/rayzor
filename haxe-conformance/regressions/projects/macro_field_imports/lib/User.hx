package lib;

import lib.Dbg.note;
import lib.Dbg.note in say;
import lib.Dbg.level in lv;

class User {
	public static function run():String
		return say("hello") + note("x") + lv;
}
