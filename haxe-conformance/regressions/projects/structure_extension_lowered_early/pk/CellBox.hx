package pk;

import pk.Handles;

interface CellBox<T> {
	var handle(default, null):Handle;
	function get():T;
}
