@:derive(Default)
class ServerConfig {
    @:default(8080)
    public var port:Int;

    @:default(60.0)
    public var timeout:Float;

    @:default(true)
    public var verbose:Bool;

    // No @:default — should use type default (0)
    public var retries:Int;

    public function new() {}
}

class Main {
    static function main() {
        var c = new ServerConfig();
        trace(c.port);     // 8080
        trace(c.timeout);  // 60
        trace(c.retries);  // 0
        trace(c.verbose);  // true
        trace("done");
    }
}
