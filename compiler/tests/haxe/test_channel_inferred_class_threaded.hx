import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;

@:derive([Send])
class Msg {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
}

// The SIGSEGV guard: an inferred `new Channel(n)` carrying class references,
// received on a spawned thread. The naive int-unbox fix crashed here because the
// class receive also erases to i64; the tag-aware unbox must recover the ref.
class Main {
    static function main() {
        var channel = new Arc(new Channel(10));
        var rx = channel.clone();
        var recv = Thread.spawn(() -> {
            var sum = 0;
            var i = 0;
            while (i < 5) {
                var m:Msg = rx.get().receive();
                sum = sum + m.v;
                i++;
            }
            return sum;
        });
        var k = 0;
        while (k < 5) {
            channel.get().send(new Msg(k * 10));
            k++;
        }
        var total = recv.join();
        if (total != 100) throw "inferred-class-threaded sum = " + total + " (expected 100)";
        Sys.println("PASS channel-inferred-class-threaded");
    }
}
