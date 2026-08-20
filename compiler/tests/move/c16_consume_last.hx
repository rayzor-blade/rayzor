@:safety @:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    @:consume
    public function close():Int { return id; }
    public function decode():Int { return id; }
}
class c16_consume_last {
    static function main():Void {
        var s = new Session(1);
        var a = s.decode();
        var b = s.decode();
        var c = s.close();    // EXPECT SILENT: nothing follows the consume
        Sys.println(a + b + c);
    }
}
