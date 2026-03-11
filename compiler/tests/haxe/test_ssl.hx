class Main {
    static function main() {
        // Test SSL Socket creation
        var sock = new sys.ssl.Socket();
        trace(sock != null);  // true

        // Test Certificate.loadDefaults — loads system CA bundle
        var certs = sys.ssl.Certificate.loadDefaults();
        trace(certs != null);  // true

        // Test Socket configuration (no network needed)
        sock.setHostname("example.com");
        trace("hostname set");

        sock.setCA(certs);
        trace("ca set");

        // Close socket
        sock.close();
        trace("done");
    }
}
