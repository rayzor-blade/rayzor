import support.SlotBase;
import support.SlotChild;
import support.SlotReader;
class Main {
  static function main() {
    var child = new SlotChild();
    var parent:SlotBase = child;
    if (parent.value(41) != 42.0) throw "imported parent parameter";
    if (child.value(41.5) != 42.5) throw "imported child parameter";
    if (parent.result(41) != 42.0) throw "imported parent result";
    if (child.result(41.5) != 42) throw "imported child result";
    if (SlotReader.value(child) != 42.0) throw "reader parent parameter";
    if (SlotReader.result(child) != 42.0) throw "reader parent result";
    if (parent.inherited(41.5) != 42.5) throw "imported inherited parameter";
    if (child.inherited(41.5) != 42.5) throw "child inherited parameter";
    Sys.println("CONFORMANCE_OK");
  }
}
