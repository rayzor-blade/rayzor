import haxe.Exception;

class Main {
    static function thrower() {
        throw new Exception("boom");
    }

    static function main() {
        try {
            thrower();
            trace("should not reach here");
        } catch (e:Exception) {
            trace(e.message);
            trace("caught");
        }
        trace("done");
    }
}
