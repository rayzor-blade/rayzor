import haxe.Exception;

class Main {
    static function main() {
        try {
            throw new Exception("boom");
        } catch(e:Exception) {
            trace(e.message);
        }
        trace("done");
    }
}
