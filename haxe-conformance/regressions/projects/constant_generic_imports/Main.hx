import support.Holder;
import support.Reader;
class Main {
  static function main() {
    var a:Holder<"imported"> = new Holder();
    var b = new Holder<"other">();
    if (a.get() != "imported" || b.get() != "other") throw "imported constants";
    var again:Holder<"imported"> = new Holder<"imported">();
    if (again.get() != a.get()) throw "imported specialization reuse";
    if (Reader.read() != "imported" || a.prefixed() != "holder imported") throw "separate module specialization";
    Sys.println("CONFORMANCE_OK");
  }
}
