import rayzor.concurrent.Thread;
import rayzor.concurrent.Channel;
import rayzor.concurrent.Arc;

class Main {
    static function main() {
        var channel = new Arc(new Channel(10));
        var threadChannel = channel.clone();

        // Thread with while loop that sends values to channel
        var sender = Thread.spawn(() -> {
            var i = 0;
            while (i < 5) {
                threadChannel.get().send(i * 10);
                i++;
            }
            return i;
        });

        var count = sender.join();
        if (count != 5) throw "sender count = " + count + " (expected 5)";

        var sum = 0;
        var j = 0;
        while (j < 5) {
            var val = channel.get().tryReceive();
            sum = sum + val;
            j++;
        }

        if (sum != 100) throw "tryReceive sum = " + sum + " (expected 100)";
        Sys.println("PASS debug-channel sum=" + sum);
    }
}
