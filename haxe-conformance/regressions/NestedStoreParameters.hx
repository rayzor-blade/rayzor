typedef ParameterRecord = {i:Int, f:Float, b:Bool, s:String, nested:{i:Int}, values:Array<Float>};
class ParameterStore {
  public var record:ParameterRecord = {i:0,f:0.0,b:false,s:"",nested:{i:0},values:[0.0]};
  public function new() {}
  public function integer(value) { record.i = value; return value; }
  public function floating(value) { this.record.f = value; return value; }
  public function boolean(value) { record.b = value; return value; }
  public function string(value) { record.s = value; return value; }
  public function nested(value) { record.nested.i = value; return value; }
  public function indexed(value) { record.values[0] = value; return value; }
  public function local(value) {
    var record:{i:String} = {i:""};
    record.i = value;
    return value;
  }
  public function parameter(record:{i:String}, value) {
    record.i = value;
    return value;
  }
}
class NestedStoreParameters {
  static function identity<T>(value:T):T return value;
  static function main() {
    var store = new ParameterStore();
    identity(store).record.i = store.integer(4);
    if (store.record.i != 4) throw "nested Int parameter";
    if (store.floating(1.5) != 1.5 || store.record.f != 1.5) throw "nested Float parameter";
    if (!store.boolean(true) || !store.record.b) throw "nested Bool parameter";
    if (store.string("record") != "record" || store.record.s != "record") throw "nested String parameter";
    if (store.nested(7) != 7 || store.record.nested.i != 7) throw "deep nested parameter";
    if (store.indexed(2.5) != 2.5 || store.record.values[0] != 2.5) throw "nested indexed parameter";
    if (store.local("local") != "local" || store.record.i != 4) throw "local receiver shadowing";
    if (store.parameter({i:""}, "parameter") != "parameter" || store.record.i != 4) throw "parameter receiver shadowing";
    if (store.integer(-4) != -4 || store.record.i != -4) throw "signed nested Int";
    if (store.floating(-2.5) != -2.5 || store.record.f != -2.5) throw "signed nested Float";
    if (store.boolean(false) != false || store.record.b != false) throw "false nested Bool";
    Sys.println("CONFORMANCE_OK");
  }
}
