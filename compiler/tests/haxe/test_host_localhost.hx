package test;

import sys.net.Host;

class Main {
    static function main() {
        var name = Host.localhost();
        trace(name);
    }
}
