class Main {
    static function traceGeneric<T>(value:T):Void {
        trace(value);
    }

    static function main() {
        traceGeneric(42);
        trace("PASS llvm-bitcast-i32-i64");
    }
}
