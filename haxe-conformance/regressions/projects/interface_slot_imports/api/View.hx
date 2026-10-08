package api;
interface View {
  public function get():Float;
  public function self():View;
  public function consume(value:Value):Void;
}
