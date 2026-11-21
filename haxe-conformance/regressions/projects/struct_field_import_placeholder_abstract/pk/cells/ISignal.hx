package pk.cells;

import pk.native.Handles;

interface ISignal<T> {
	var ptr(default, null):SignalHandle;
	function get():T;
}
