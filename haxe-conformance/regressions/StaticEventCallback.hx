class StaticEventCallback {
    static var event = haxe.MainLoop.add(function() {
        event.stop();
        Sys.println("CONFORMANCE_OK");
    });

    static function main() {}
}
