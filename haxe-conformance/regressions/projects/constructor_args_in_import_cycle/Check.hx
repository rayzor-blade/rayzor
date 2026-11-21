class Check {
	// The constructor calls come from a macro, so the import scan does not see them.
	static macro function make(code:String):haxe.macro.Expr
		return haxe.macro.Context.parse(code, haxe.macro.Context.currentPos());

	public static function run():String {
		var one:lib.Component<Dynamic> = make("new lib.RadioGroup.RadioGroupItem({value: 'small'})");
		var two:lib.Component<Dynamic> = make("new lib.RadioGroup.RadioGroupItem({value: 'large'}, [])");
		var three:lib.Component<Dynamic> = make("new lib.RadioGroup.RadioGroupItem({value: 'big'}, [], lib.Tree.current, 3)");
		return [for (c in [one, two, three]) c.given + "/" + c.count].join(" ");
	}
}
