import pkg.Holder;

class Main {
    static var failures = 0;
    static function ok(what:String, cond:Bool):Void {
        if (!cond) { Sys.println("FAIL " + what); failures++; }
    }

    static function main() {
        Holder.entry(true);
        Holder.entry(false);

        ok("direct visible after in-function store", !Holder.directIsNull());
        ok("viaCall visible after callee store", !Holder.viaCallIsNull());

        if (failures == 0) Sys.println("PASS: statics written by a callee are visible to all readers");
        else { Sys.println("FAILURES: " + failures); Sys.exit(1); }
    }
}
