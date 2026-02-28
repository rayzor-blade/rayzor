class WithGetSet {
    var _x:Int;
    public var x(get, set):Int;

    public function new(v:Int) {
        _x = v;
    }

    function get_x():Int {
        return _x;
    }

    function set_x(v:Int):Int {
        _x = v;
        return v;
    }
}

class WithReadOnly {
    var _name:String;
    public var name(get, never):String;

    public function new(n:String) {
        _name = n;
    }

    function get_name():String {
        return _name;
    }
}

class Main {
    static function main() {
        // Test get/set property with accessor methods
        var obj = new WithGetSet(10);
        trace(obj.x);        // 10 (calls get_x)
        obj.x = 42;          // calls set_x
        trace(obj.x);        // 42

        // Test read-only property
        var ro = new WithReadOnly("hello");
        trace(ro.name);      // hello (calls get_name)

        trace("done");
    }
}
