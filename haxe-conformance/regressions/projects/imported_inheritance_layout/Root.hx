class Root {
    public var count:Int;
    public var text:String;
    public var doubled(get, never):Int;
    function get_doubled():Int return count * 2;
    public function new(value:Int) {
        count = value;
        text = "root";
    }
    public static function make(value:Int):Leaf return new Leaf(value);
}
