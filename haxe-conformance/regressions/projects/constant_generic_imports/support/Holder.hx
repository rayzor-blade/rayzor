package support;
import utilities.Prefix;
@:generic class Holder<@:const VALUE> {
  public function new() {}
  public function get() return VALUE;
  public function prefixed() return Prefix.get() + " " + VALUE;
}
