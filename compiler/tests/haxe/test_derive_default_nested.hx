@:derive(Default)
class Size {
    public var width:Int;
    public var height:Int;

    public function new() {}
}

@:derive(Default)
class Config {
    public var name:String;
    public var size:Size;
    public var scale:Float;

    public function new() {}
}

// Has a no-arg constructor but NO @:derive(Default)
class Logger {
    public var level:Int;

    public function new() {
        this.level = 42;
    }
}

@:derive(Default)
class App {
    public var config:Config;
    public var logger:Logger;
    public var enabled:Bool;

    public function new() {}
}

class Main {
    static function main() {
        var app = new App();

        // Config should be recursively default-initialized
        trace(app.config.scale);
        trace(app.config.name);

        // Size inside Config should also be recursively default-initialized
        trace(app.config.size.width);
        trace(app.config.size.height);

        // Logger should be constructed via no-arg constructor (fallback)
        trace(app.logger.level);

        trace(app.enabled);
    }
}
