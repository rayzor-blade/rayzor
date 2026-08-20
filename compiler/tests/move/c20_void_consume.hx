@:safety @:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    // `session.close();` in statement position is how the feature is written.
    @:consume
    public function close():Void { Sys.println(id); }
    public function decode():Int { return id; }
}
class c20_void_consume {
    static function main():Void {
        var s = new Session(1);
        s.close();
        Sys.println(s.decode());   // EXPECT ERROR: close() ended the receiver
    }
}
