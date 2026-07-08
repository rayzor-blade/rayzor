import rayzor.concurrent.SpinPool;

class Main {
    static function main() {
        var pool = new SpinPool(2);

        pool.parallelRows(4, function(lo:Int, hi:Int, node:Int):Void {});
        pool.parallelRows(4, function(lo:Int, hi:Int, node:Int):Void {});
        pool.shutdown();

        trace("PASS spinpool-parallel-rows-cranelift");
    }
}
