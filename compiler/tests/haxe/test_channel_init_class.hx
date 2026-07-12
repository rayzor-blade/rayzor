import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;

@:derive([Send])
class Msg {
    public var v:Int;
    public function new(v:Int) { this.v = v; }
}

// The Channel.init<T> static constructor path (a separate code route into the
// same handle) carrying class references across a spawned thread.
class Main {
    static function main() {
        var channel = new Arc(Channel.init(10));
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
        if (total != 100) throw "init-class sum = " + total + " (expected 100)";
        Sys.println("PASS channel-init-class");
    }
}
