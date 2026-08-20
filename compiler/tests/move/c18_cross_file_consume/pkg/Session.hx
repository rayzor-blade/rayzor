package pkg;

@:safety @:move
class Session {
    public var id:Int;
    public function new(id:Int) { this.id = id; }
    @:consume
    public function close():Int { return id; }
    public function decode():Int { return id; }
}
