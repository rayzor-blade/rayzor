class TestStartupHooksOnce {
    static var initialized:Bool = initialize();

    static function initialize():Bool {
        var current = Sys.getEnv("RAYZOR_STARTUP_HOOK_COUNT");
        var count = current == null ? 0 : Std.parseInt(current);
        Sys.putEnv("RAYZOR_STARTUP_HOOK_COUNT", Std.string(count + 1));
        return true;
    }

    static function main() {
        var startupCount = Std.parseInt(Sys.getEnv("RAYZOR_STARTUP_HOOK_COUNT"));
        if (!initialized || startupCount != 1) {
            trace('FAIL: startup hooks ran $startupCount times');
            return;
        }
        trace("PASS");
    }
}
