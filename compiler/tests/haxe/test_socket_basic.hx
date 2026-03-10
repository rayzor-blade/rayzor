package test;

import sys.net.Socket;
import sys.net.Host;

class Main {
    static function main() {
        var server = new Socket();
        var host = new Host("127.0.0.1");
        server.bind(host, 19876);
        server.listen(1);

        var client = new Socket();
        client.connect(new Host("127.0.0.1"), 19876);

        var conn = server.accept();

        client.write("hello");
        var data = conn.read();

        conn.close();
        client.close();
        server.close();
    }
}
