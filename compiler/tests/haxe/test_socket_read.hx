import sys.net.Host;
import sys.net.Socket;

class TestSocketRead {
    static function main() {
        var port = 19883;
        var server = new Socket();
        server.setTimeout(5.0);
        server.bind(new Host("127.0.0.1"), port);
        server.listen(1);

        var client = new Socket();
        client.setTimeout(5.0);
        client.connect(new Host("127.0.0.1"), port);

        var conn = server.accept();
        if (conn == null) throw "accept returned null";
        conn.setTimeout(5.0);

        client.write("ping");
        client.shutdown(false, true);

        var got = conn.read();
        if (got.length != 4) throw "bad read length " + got.length;
        if (got.charCodeAt(0) != 112) throw "bad byte 0";
        if (got.charCodeAt(1) != 105) throw "bad byte 1";
        if (got.charCodeAt(2) != 110) throw "bad byte 2";
        if (got.charCodeAt(3) != 103) throw "bad byte 3";

        conn.close();
        client.close();
        server.close();
        Sys.println("socket read ok");
    }
}
