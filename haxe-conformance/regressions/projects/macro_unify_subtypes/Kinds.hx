interface IBase {}
interface IChild extends IBase {}

class Base implements IBase {
	public function new() {}
}

class Sub extends Base {}

class ViaChild implements IChild {
	public function new() {}
}

class Unrelated {
	public function new() {}
}
