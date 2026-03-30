/* tslint:disable */
/* eslint-disable */

export function rayzor_window_create(title: string, x: number, y: number, w: number, h: number, style: number): number;

export function rayzor_window_create_centered(title: string, w: number, h: number): number;

export function rayzor_window_destroy(h: number): void;

export function rayzor_window_event_button(h: number, idx: number): number;

export function rayzor_window_event_count(h: number): number;

export function rayzor_window_event_height(h: number, idx: number): number;

export function rayzor_window_event_key(h: number, idx: number): number;

export function rayzor_window_event_modifiers(h: number, idx: number): number;

export function rayzor_window_event_scroll_x(h: number, idx: number): number;

export function rayzor_window_event_scroll_y(h: number, idx: number): number;

export function rayzor_window_event_type(h: number, idx: number): number;

export function rayzor_window_event_width(h: number, idx: number): number;

export function rayzor_window_event_x(h: number, idx: number): number;

export function rayzor_window_event_y(h: number, idx: number): number;

export function rayzor_window_get_display_handle(_h: number): number;

export function rayzor_window_get_handle(h: number): number;

export function rayzor_window_get_height(h: number): number;

export function rayzor_window_get_mouse_x(h: number): number;

export function rayzor_window_get_mouse_y(h: number): number;

export function rayzor_window_get_width(h: number): number;

export function rayzor_window_get_x(_h: number): number;

export function rayzor_window_get_y(_h: number): number;

export function rayzor_window_is_focused(h: number): number;

export function rayzor_window_is_fullscreen(_h: number): number;

export function rayzor_window_is_key_down(h: number, key: number): number;

export function rayzor_window_is_minimized(_h: number): number;

export function rayzor_window_is_mouse_down(h: number, button: number): number;

export function rayzor_window_is_visible(h: number): number;

export function rayzor_window_poll_events(h: number): number;

/**
 * Run a frame-driven render loop using requestAnimationFrame.
 * `callback` is a JS function that returns true to continue, false to stop.
 * On each frame: poll events → call callback → request next frame.
 */
export function rayzor_window_run_loop(win_h: number, callback: Function): void;

export function rayzor_window_set_floating(_h: number, _on_top: number): void;

export function rayzor_window_set_fullscreen(h: number, fs: number): void;

export function rayzor_window_set_max_size(_h: number, _w: number, _ht: number): void;

export function rayzor_window_set_min_size(_h: number, _w: number, _ht: number): void;

export function rayzor_window_set_opacity(h: number, opacity: number): void;

export function rayzor_window_set_position(_h: number, _x: number, _y: number): void;

export function rayzor_window_set_size(h: number, w: number, ht_val: number): void;

export function rayzor_window_set_title(_h: number, title: string): void;

export function rayzor_window_set_visible(h: number, vis: number): void;

export function rayzor_window_was_resized(h: number): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly rayzor_window_create: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly rayzor_window_create_centered: (a: number, b: number, c: number, d: number) => number;
  readonly rayzor_window_destroy: (a: number) => void;
  readonly rayzor_window_event_button: (a: number, b: number) => number;
  readonly rayzor_window_event_count: (a: number) => number;
  readonly rayzor_window_event_height: (a: number, b: number) => number;
  readonly rayzor_window_event_key: (a: number, b: number) => number;
  readonly rayzor_window_event_modifiers: (a: number, b: number) => number;
  readonly rayzor_window_event_scroll_x: (a: number, b: number) => number;
  readonly rayzor_window_event_scroll_y: (a: number, b: number) => number;
  readonly rayzor_window_event_type: (a: number, b: number) => number;
  readonly rayzor_window_event_width: (a: number, b: number) => number;
  readonly rayzor_window_event_x: (a: number, b: number) => number;
  readonly rayzor_window_event_y: (a: number, b: number) => number;
  readonly rayzor_window_get_display_handle: (a: number) => number;
  readonly rayzor_window_get_handle: (a: number) => number;
  readonly rayzor_window_get_height: (a: number) => number;
  readonly rayzor_window_get_mouse_x: (a: number) => number;
  readonly rayzor_window_get_mouse_y: (a: number) => number;
  readonly rayzor_window_get_width: (a: number) => number;
  readonly rayzor_window_get_x: (a: number) => number;
  readonly rayzor_window_get_y: (a: number) => number;
  readonly rayzor_window_is_focused: (a: number) => number;
  readonly rayzor_window_is_fullscreen: (a: number) => number;
  readonly rayzor_window_is_key_down: (a: number, b: number) => number;
  readonly rayzor_window_is_minimized: (a: number) => number;
  readonly rayzor_window_is_mouse_down: (a: number, b: number) => number;
  readonly rayzor_window_is_visible: (a: number) => number;
  readonly rayzor_window_poll_events: (a: number) => number;
  readonly rayzor_window_run_loop: (a: number, b: any) => void;
  readonly rayzor_window_set_floating: (a: number, b: number) => void;
  readonly rayzor_window_set_fullscreen: (a: number, b: number) => void;
  readonly rayzor_window_set_max_size: (a: number, b: number, c: number) => void;
  readonly rayzor_window_set_min_size: (a: number, b: number, c: number) => void;
  readonly rayzor_window_set_opacity: (a: number, b: number) => void;
  readonly rayzor_window_set_position: (a: number, b: number, c: number) => void;
  readonly rayzor_window_set_size: (a: number, b: number, c: number) => void;
  readonly rayzor_window_set_title: (a: number, b: number, c: number) => void;
  readonly rayzor_window_set_visible: (a: number, b: number) => void;
  readonly rayzor_window_was_resized: (a: number) => number;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke______: (a: number, b: number) => void;
  readonly wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut_____Output_______: (a: number, b: number) => void;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wasm_bindgen_335648ada7beb221___JsValue_____: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut__wasm_bindgen_335648ada7beb221___JsValue____Output_______: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
