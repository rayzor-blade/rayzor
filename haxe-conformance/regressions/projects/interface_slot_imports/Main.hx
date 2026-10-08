import api.View;
import api.Value;
import api.Reader;
import other.Owner;
class Main {
  static function main() {
    var object = new Value();
    var view:View = object;
    if (view.get() != 42.0) throw "imported numeric result";
    var returned:View = view.self();
    if (returned.get() != 42.0) throw "imported reference result";
    view.consume(object);
    if (Reader.get(view) != 42.0) throw "reader numeric result";
    if (Reader.self(view).get() != 42.0) throw "reader reference result";
    Reader.consume(view, object);
    var separate:other.View = new Owner();
    if (separate.count() != 3) throw "same-named interface in another package";
    Sys.println("CONFORMANCE_OK");
  }
}
