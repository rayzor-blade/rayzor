import rayzor.concurrent.Channel;

// Explicit Channel<Int>, same thread: the always-worked path — must stay green
// (receive() and tryReceive() both unbox the DynamicValue to the concrete Int).
class Main {
    static function main() {
        var ch = new Channel<Int>(10);
        ch.send(111);
        ch.send(222);
        var a:Int = ch.receive();
        if (a != 111) throw "explicit-int recvA = " + a;
        var b:Int = ch.receive();
        if (b != 222) throw "explicit-int recvB = " + b;
        // Channel now empty: tryReceive is null.
        var c:Null<Int> = ch.tryReceive();
        if (c != null) throw "explicit-int empty tryReceive was not null";
        Sys.println("PASS channel-explicit-int");
    }
}
