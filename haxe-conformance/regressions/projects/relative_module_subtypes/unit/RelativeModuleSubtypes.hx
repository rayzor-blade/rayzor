package unit;

class RelativeModuleSubtypes extends Workers.GenericChecks {
    function checkGenericCalls() {
        var worker = new Workers.Worker();
        if (!same(cast(worker, Workers.Parent), worker)) throw "generic interface identity";
        var child = cast(worker, Workers.Child);
        if (!same(cast(child, Workers.Parent), child)) throw "generic parent interface identity";
        if (!same("prefix" + "value", "prefixvalue")) throw "generic String equality";
        if (!same(1.25, 1.25) || same(1.25, 2.5)) throw "generic numeric equality";
        var result:Float = identity(2.5);
        if (result != 2.5) throw "generic numeric result";
    }

    static function main() {
        var worker = new Workers.Worker();
        if (worker.value() != 42) throw "relative subtype constructor";
        var parent = cast(worker, Workers.Parent);
        var child = cast(worker, Workers.Child);
        if (parent.value() != 42 || child.value() != 42) throw "relative interface casts";
        var caught = false;
        try cast(worker, Workers.Other) catch (_:Dynamic) caught = true;
        if (!caught) throw "incompatible interface cast";
        new RelativeModuleSubtypes().checkGenericCalls();
        Sys.println("CONFORMANCE_OK");
    }
}
