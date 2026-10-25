// A target passed to a call is finalised once nothing holds it, though a
// copy of its handle may remain in a frame the call left behind.
class GcFinalizerAfterCall {
    static function main():Void {
        var called = false;
        var held:Null<String> = null;
        var finalizer = new haxe.GcFinalizer(function(v:String) {
            called = true;
            held = v;
        });
        finalizer.register({id: 1}, "collected");
        eval.vm.Gc.full_major();
        eval.vm.Gc.full_major();
        if (!called || held != "collected") throw "finaliser did not run";
        Sys.println("CONFORMANCE_OK");
    }
}
