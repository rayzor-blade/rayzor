@:safety @:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    @:consume
    public function close():Int { return id; }
    public function decode():Int { return id; }
}
class c15_consume_receiver {
    static function main():Void {
        var s = new Session(1);
        var a = s.close();
        var b = s.decode();   // EXPECT ERROR: close() ends the receiver
        Sys.println(a + b);
    }
}
