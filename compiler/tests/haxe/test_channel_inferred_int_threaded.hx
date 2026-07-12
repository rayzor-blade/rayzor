import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;

// The primary bug: an inferred `new Channel(n)` carrying primitives, received on
// a spawned thread, erases T to i64 — the old raw reinterpret returned garbage.
class Main {
    static function main() {
        var channel = new Arc(new Channel(10));
        var rx = channel.clone();
        var recv = Thread.spawn(() -> {
            var sum = 0;
            var i = 0;
            while (i < 5) {
                sum = sum + rx.get().receive();
                i++;
            }
            return sum;
        });
        var k = 0;
        while (k < 5) {
            channel.get().send(k * 10);
            k++;
        }
        var total = recv.join();
        if (total != 100) throw "inferred-int-threaded sum = " + total + " (expected 100)";
        Sys.println("PASS channel-inferred-int-threaded");
    }
}
