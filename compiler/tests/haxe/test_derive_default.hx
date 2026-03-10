@:derive(Default)
class Config {
    public var width:Int;
    public var height:Int;
    public var scale:Float;
    public var enabled:Bool;

    public function new() {}
}

class Main {
    static function main() {
        var c = new Config();
        trace(c.width);
        trace(c.height);
        trace(c.scale);
        trace(c.enabled);
    }
}
