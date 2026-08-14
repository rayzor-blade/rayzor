class TestInterpreterSysGetEnv {
    static var bootEnv:String = Sys.getEnv("RAYZOR_INTERP_TEST_ENV");

    static function main() {
        var runEnv = Sys.getEnv("RAYZOR_INTERP_TEST_ENV");
        if (bootEnv != "ok" || runEnv != "ok") {
            trace("FAIL interpreter-sys-getenv boot=" + bootEnv + " run=" + runEnv);
            Sys.exit(1);
        }
        trace("PASS interpreter-sys-getenv");
    }
}
