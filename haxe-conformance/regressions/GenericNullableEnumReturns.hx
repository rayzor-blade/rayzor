private enum Item { Entry(text:String); }
class GenericNullableEnumReturns {
  static function main() {
    var list = new haxe.ds.List<Item>();
    list.add(Entry("payload"));
    if (Std.string(list.first()) != "Entry(payload)") throw "first: " + Std.string(list.first());
    if (Std.string(list.last()) != "Entry(payload)") throw "last: " + Std.string(list.last());
    if (Std.string(list.pop()) != "Entry(payload)") throw "pop";
    if (list.first() != null) throw "empty";
    var items = [Entry("matched")];
    var text = switch items {
      case [Entry(value)]: value;
      default: throw "array pattern";
    };
    if (text != "matched") throw "enum payload";
    Sys.println("CONFORMANCE_OK");
  }
}
