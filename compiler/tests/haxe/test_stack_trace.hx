import haxe.Exception;

class Main {
    static function thrower() {
        throw new Exception("boom");
    }

    static function main() {
        try {
            thrower();
        } catch(e:Exception) {
            trace(e.message);
        }
        trace("done");
    }
}
