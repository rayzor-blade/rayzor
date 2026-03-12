// Test Option<T> generic enum
// Expected output:
// 42
// none

import haxe.ds.Option;

class Main {
    static function main() {
        var x:Option<Int> = Some(42);
        switch(x) {
            case Some(v): trace(v);
            case None: trace("none");
        }

        var y:Option<Int> = None;
        switch(y) {
            case Some(v): trace(v);
            case None: trace("none");
        }
    }
}
