package utest;

/** Minimal stand-in for utest's Async.

    utest hands a test method that takes one an Async, then keeps the runner's
    event loop turning until `done()` is called -- on a threaded target, the
    main thread's `haxe.EventLoop`. The harness does the same through `wait()`,
    so callbacks a test schedules on the main loop (timers, `run`, thread
    results posted back) run where they would under utest. A branch is a child
    Async; a parent with branches resolves when all of them have. */
class Async {
    var resolved = false;
    var branches:Array<Async> = [];
    var timeoutMs:Int;

    public function new(timeoutMs:Int = 5000) {
        this.timeoutMs = timeoutMs;
    }

    public function done(?pos:haxe.PosInfos):Void {
        resolved = true;
    }

    public function setTimeout(timeoutMs:Int, ?pos:haxe.PosInfos):Void {
        this.timeoutMs = timeoutMs;
    }

    public function branch(?f:Async->Void, ?pos:haxe.PosInfos):Async {
        var child = new Async(timeoutMs);
        branches.push(child);
        if (f != null) f(child);
        return child;
    }

    function isResolved():Bool {
        if (resolved) return true;
        if (branches.length == 0) return false;
        for (b in branches) if (!b.isResolved()) return false;
        return true;
    }

    /** Called by the harness after the test method returns. */
    public function wait():Void {
        var deadline = haxe.Timer.stamp() + timeoutMs / 1000;
        while (!isResolved() && haxe.Timer.stamp() < deadline) {
            haxe.EventLoop.main.loopOnce();
            Sys.sleep(0.001);
        }
        if (!isResolved()) unit.ConfCheck.fail("async test timed out");
    }
}
