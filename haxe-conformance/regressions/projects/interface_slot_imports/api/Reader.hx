package api;
class Reader {
  public static function get(view:View):Float return view.get();
  public static function self(view:View):View return view.self();
  public static function consume(view:View, value:Value):Void view.consume(value);
}
