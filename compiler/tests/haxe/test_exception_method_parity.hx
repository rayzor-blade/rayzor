import haxe.Exception;

class Main {
    static function thrower() {
        throw new Exception("boom");
    }

    static function main() {
        try {
            thrower();
        } catch (e:Exception) {
            trace("message=" + e.message);
            trace("toString=" + e.toString());
            var details = e.details();
            trace("details=" + details);
            trace("hasMessage=" + (details.indexOf("boom") >= 0));
            trace("hasStack=" + (details.indexOf("Called from") >= 0));
        }
    }
}
