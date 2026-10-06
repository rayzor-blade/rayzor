import same.Worker;
import foreign.Error as ForeignError;

class PackageSiblings {
    static function main() {
        if (new ForeignError().value != 17) throw "foreign class";
        if (Worker.code() != 31) throw "same-package enum";
        if (same.ImportedWorker.code() != 17) throw "explicit import precedence";
        if (same.LocalWorker.code() != 23) throw "local binding precedence";
        Sys.println("CONFORMANCE_OK");
    }
}
