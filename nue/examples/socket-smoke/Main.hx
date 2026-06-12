import haxe.io.Bytes;
import sys.net.Host;
import sys.net.Socket;

class Main {
    static function main() {
        var port = 19878;
        var host = new Host("127.0.0.1");
        var server = new Socket();
        server.setTimeout(2.0);
        server.bind(host, port);
        server.listen(1);

        var client = new Socket();
        client.setTimeout(2.0);
        client.setFastSend(true);
        client.connect(new Host("127.0.0.1"), port);

        var conn = server.accept();
        if (conn == null) throw "accept returned null";
        conn.setTimeout(2.0);
        conn.setFastSend(true);

        client.output.writeString("ping");
        client.output.flush();

        var inBuf = Bytes.alloc(4);
        var gotN = conn.input.readBytes(inBuf, 0, 4);
        if (gotN != 4
            || inBuf.get(0) != 112
            || inBuf.get(1) != 105
            || inBuf.get(2) != 110
            || inBuf.get(3) != 103) {
            throw "server expected ping";
        }

        conn.output.writeString("pong");
        conn.output.flush();

        var outBuf = Bytes.alloc(4);
        var replyN = client.input.readBytes(outBuf, 0, 4);
        if (replyN != 4
            || outBuf.get(0) != 112
            || outBuf.get(1) != 111
            || outBuf.get(2) != 110
            || outBuf.get(3) != 103) {
            throw "client expected pong";
        }

        conn.close();
        client.close();
        server.close();
        trace("socket echo ok");
    }
}
