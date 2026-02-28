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
        trace(count);

        var sum = 0;
        var j = 0;
        while (j < 5) {
            var val = channel.get().tryReceive();
            sum = sum + val;
            j++;
        }

        trace(sum);
        trace("done");
    }
}
