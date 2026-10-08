enum RepresentationEnum {
  Empty;
  Payload(value:Int);
}
class EnumValueRepresentation {
  static function identity(value:EnumValue):EnumValue return value;
  static function main() {
    var empty:EnumValue = RepresentationEnum.Empty;
    var payload:EnumValue = RepresentationEnum.Payload(42);
    var returnedEmpty:RepresentationEnum = cast identity(empty);
    switch (returnedEmpty) {
      case Empty:
      default: throw "constant enum value";
    }
    var returnedPayload:RepresentationEnum = cast identity(payload);
    switch (returnedPayload) {
      case Payload(value): if (value != 42) throw "enum value parameter";
      default: throw "enum value pointer";
    }
    Sys.println("CONFORMANCE_OK");
  }
}
