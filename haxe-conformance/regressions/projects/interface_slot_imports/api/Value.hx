package api;
class Value implements View {
  public function new() {}
  public function get():Int return 42;
  public function self():Value return this;
  public function consume(value:View):Void {
    if (value.get() != 42.0) throw "imported interface parameter";
  }
}
