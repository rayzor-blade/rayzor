package t;

@:build(t.KeysMac.build("t.Sizes.SizesData"))
enum abstract Size(String) {
	var Small = "small";
	var Large = "large";
}
