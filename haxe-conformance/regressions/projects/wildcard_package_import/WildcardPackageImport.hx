import lib.pk.Other;

class WildcardPackageImport {
    static function main() {
        if (app.User.run() != 19) throw "run " + app.User.run();
        if (Other.value() != 7) throw "other";
        trace("CONFORMANCE_OK");
    }
}
