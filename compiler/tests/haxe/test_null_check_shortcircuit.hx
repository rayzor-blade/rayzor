// `s != null && ...` on a reference lowers the short-circuit through a Not
// applied to the pointer itself. The LLVM JIT asked that value for its integer
// variant and inkwell panicked, so any function doing a null test before using
// a String failed to compile rather than failing to run.
class Main {
    static function main():Void {
        check(null, "null");
        check("", "empty");
        check("abc", "abc");
    }

    static function check(s:String, tag:String):Void {
        if (s != null && s.length > 0) {
            Sys.println(tag + " -> non-empty, len " + s.length);
        } else if (s != null) {
            Sys.println(tag + " -> empty");
        } else {
            Sys.println(tag + " -> null");
        }
        if (!(s == null)) {
            Sys.println(tag + " -> not null");
        }
    }
}
