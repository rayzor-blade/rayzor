import sys.net.Socket;
import sys.net.Host;

class Main {
    static function main() {
        var host = Host.localhost();
        trace(host);
        var s = new Socket();
        trace("socket created");
        s.close();
        trace("socket closed");
    }
}
