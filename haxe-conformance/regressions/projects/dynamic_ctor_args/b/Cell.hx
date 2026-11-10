package b;

class Cell<T> {
	var value:T;

	public function new(v:T)
		value = v;

	public function get():T
		return value;
}

// The constructor names only the second parameter.
class Tagged<K, V> {
	var value:V;

	public function new(v:V)
		value = v;

	public function get():V
		return value;
}
