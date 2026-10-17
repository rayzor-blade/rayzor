import haxe.Exception;
import haxe.coro.IContinuation;
import haxe.coro.SuspensionResult;
import haxe.coro.context.Context;

private class Tracking<T> implements IContinuation<T> {
    public var context(get, never):Context;
    public var resumes = 0;
    public var result:Null<T> = null;
    public var error:Null<Exception> = null;
    public function new() {}
    function get_context():Context return Context.empty;
    public function resume(result:Null<T>, error:Null<Exception>):Void {
        resumes++;
        this.result = result;
        this.error = error;
    }
}

private class Suspender<T> {
    public var cont:Null<IContinuation<T>> = null;
    public function new() {}
    @:coroutine(transformed)
    public function suspend(cont:IContinuation<T>):SuspensionResult<T> {
        this.cont = cont;
        return new SuspensionResult(Pending);
    }
}

private class Counter {
    var i = 0;
    public function new() {}
    @:coroutine public function hasNext():Bool return i < 3;
    public function next():Int return i++;
}

class CoroutineFibers {
    static function main() {
        final sus = new Suspender<Int>();
        @:coroutine function suspending():Int {
            final v = sus.suspend();
            return v + 1;
        }
        var c = new Tracking<Int>();
        suspending(c);
        if (c.resumes != 0) throw "resumed before the suspension ended";
        sus.cont.resume(41, null);
        if (c.resumes != 1 || c.result != 42) throw "resumed value";

        @:coroutine function inner():Int return 20;
        @:coroutine function outer():Int return inner() + inner() + 2;
        var c2 = new Tracking<Int>();
        outer(c2);
        if (c2.resumes != 1 || c2.result != 42) throw "nested coroutine calls";

        @:coroutine function items():Array<Int> {
            final ret = [];
            for (v in new Counter()) ret.push(v);
            return ret;
        }
        var c3 = new Tracking<Array<Int>>();
        items(c3);
        if (c3.result.length != 3 || c3.result[2] != 2) throw "async iterator loop";

        @:coroutine function fails():Int throw "boom";
        var c4 = new Tracking<Int>();
        fails(c4);
        if (c4.resumes != 1 || c4.error == null) throw "error reaches the continuation";
        Sys.println("CONFORMANCE_OK");
    }
}
