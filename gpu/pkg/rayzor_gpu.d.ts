/* tslint:disable */
/* eslint-disable */

export function rayzor_gpu_compute_abs(dev: number, a: number): number;

export function rayzor_gpu_compute_add(dev: number, a: number, b: number): number;

export function rayzor_gpu_compute_alloc_buffer(dev_h: number, numel: number, dtype: number): number;

export function rayzor_gpu_compute_buffer_dtype(buf_h: number): number;

export function rayzor_gpu_compute_buffer_numel(buf_h: number): number;

export function rayzor_gpu_compute_create(): Promise<number>;

export function rayzor_gpu_compute_destroy(h: number): void;

export function rayzor_gpu_compute_div(dev: number, a: number, b: number): number;

export function rayzor_gpu_compute_dot(dev_h: number, a_h: number, b_h: number): number;

export function rayzor_gpu_compute_exp(dev: number, a: number): number;

export function rayzor_gpu_compute_free_buffer(_dev_h: number, buf_h: number): void;

export function rayzor_gpu_compute_gelu(dev: number, a: number): number;

export function rayzor_gpu_compute_is_available(): number;

export function rayzor_gpu_compute_log(dev: number, a: number): number;

export function rayzor_gpu_compute_matmul(dev_h: number, a_h: number, b_h: number, m: number, k: number, n: number): number;

export function rayzor_gpu_compute_max(dev_h: number, buf_h: number): number;

export function rayzor_gpu_compute_mean(dev_h: number, buf_h: number): number;

export function rayzor_gpu_compute_min(dev_h: number, buf_h: number): number;

export function rayzor_gpu_compute_mul(dev: number, a: number, b: number): number;

export function rayzor_gpu_compute_neg(dev: number, a: number): number;

export function rayzor_gpu_compute_relu(dev: number, a: number): number;

export function rayzor_gpu_compute_sigmoid(dev: number, a: number): number;

export function rayzor_gpu_compute_silu(dev: number, a: number): number;

export function rayzor_gpu_compute_sqrt(dev: number, a: number): number;

export function rayzor_gpu_compute_sub(dev: number, a: number, b: number): number;

export function rayzor_gpu_compute_sum(dev_h: number, buf_h: number): number;

export function rayzor_gpu_compute_tanh(dev: number, a: number): number;

export function rayzor_gpu_gfx_buffer_create(dev_h: number, size: number, usage: number): number;

export function rayzor_gpu_gfx_buffer_create_with_data(dev_h: number, data: Uint8Array, usage: number): number;

export function rayzor_gpu_gfx_buffer_destroy(h: number): void;

export function rayzor_gpu_gfx_buffer_from_bytes(dev_h: number, data: Uint8Array, usage_flags: number): number;

export function rayzor_gpu_gfx_buffer_write(buf_h: number, dev_h: number, offset: number, data: Uint8Array): void;

export function rayzor_gpu_gfx_buffer_write_bytes(buf_h: number, dev_h: number, offset: number, data: Uint8Array): void;

export function rayzor_gpu_gfx_cmd_begin_pass(cmd_h: number, color_view_h: number, load_op: number, clear_r: number, clear_g: number, clear_b: number, clear_a: number, depth_view_h: number): void;

export function rayzor_gpu_gfx_cmd_begin_pass_mrt(cmd_h: number, _count: number, _color_views: Int32Array, _load_ops: Int32Array, _clear_colors: Float64Array, _depth_h: number): void;

export function rayzor_gpu_gfx_cmd_create(): number;

export function rayzor_gpu_gfx_cmd_destroy(h: number): void;

export function rayzor_gpu_gfx_cmd_draw(cmd_h: number, vertex_count: number, instance_count: number, first_vertex: number, first_instance: number): void;

export function rayzor_gpu_gfx_cmd_draw_indexed(cmd_h: number, index_count: number, instance_count: number, first_index: number, base_vertex: number, first_instance: number): void;

export function rayzor_gpu_gfx_cmd_end_pass(cmd_h: number): void;

export function rayzor_gpu_gfx_cmd_set_bind_group(cmd_h: number, group_index: number, bind_group_h: number): void;

export function rayzor_gpu_gfx_cmd_set_index_buffer(cmd_h: number, buffer_h: number, format: number): void;

export function rayzor_gpu_gfx_cmd_set_pipeline(cmd_h: number, pipeline_h: number): void;

export function rayzor_gpu_gfx_cmd_set_scissor(cmd_h: number, x: number, y: number, w: number, h: number): void;

export function rayzor_gpu_gfx_cmd_set_vertex_buffer(cmd_h: number, slot: number, buffer_h: number): void;

export function rayzor_gpu_gfx_cmd_set_viewport(cmd_h: number, x: number, y: number, w: number, h: number, min_depth: number, max_depth: number): void;

export function rayzor_gpu_gfx_cmd_submit(cmd_h: number, dev_h: number): void;

export function rayzor_gpu_gfx_device_create(): Promise<number>;

export function rayzor_gpu_gfx_device_destroy(h: number): void;

export function rayzor_gpu_gfx_is_available(): number;

export function rayzor_gpu_gfx_pipeline_add_color_target(pipe_h: number, format: number): void;

export function rayzor_gpu_gfx_pipeline_add_layout(builder_h: number, layout_h: number): void;

export function rayzor_gpu_gfx_pipeline_begin(): number;

export function rayzor_gpu_gfx_pipeline_build(pipe_h: number, dev_h: number): number;

export function rayzor_gpu_gfx_pipeline_destroy(h: number): void;

export function rayzor_gpu_gfx_pipeline_set_cull(pipe_h: number, cull: number): void;

export function rayzor_gpu_gfx_pipeline_set_depth_simple(builder_h: number, depth_format: number): void;

export function rayzor_gpu_gfx_pipeline_set_format(pipe_h: number, format: number): void;

export function rayzor_gpu_gfx_pipeline_set_shader(pipe_h: number, shader_h: number): void;

export function rayzor_gpu_gfx_pipeline_set_topology(pipe_h: number, topo: number): void;

export function rayzor_gpu_gfx_pipeline_set_vertex_layout_simple(builder_h: number, stride: number, attr_count: number, attr_data: Int32Array): void;

export function rayzor_gpu_gfx_sampler_create(dev_h: number, min: number, mag: number, addr: number): number;

export function rayzor_gpu_gfx_sampler_destroy(h: number): void;

export function rayzor_gpu_gfx_shader_create_hx(dev_h: number, wgsl: string, vs: string, fs: string): number;

export function rayzor_gpu_gfx_shader_destroy(h: number): void;

export function rayzor_gpu_gfx_surface_create(dev_h: number, _window_handle: number, _display_handle: number, width: number, height: number): number;

export function rayzor_gpu_gfx_surface_create_canvas(dev_h: number, canvas_id: string, width: number, height: number): number;

export function rayzor_gpu_gfx_surface_destroy(h: number): void;

export function rayzor_gpu_gfx_surface_get_format(h: number): number;

export function rayzor_gpu_gfx_surface_get_texture(h: number): number;

export function rayzor_gpu_gfx_surface_present(h: number): void;

export function rayzor_gpu_gfx_surface_resize(h: number, dev_h: number, w: number, ht_val: number): void;

export function rayzor_gpu_gfx_texture_create(dev_h: number, w: number, h: number, fmt: number, usage: number): number;

export function rayzor_gpu_gfx_texture_destroy(h: number): void;

export function rayzor_gpu_gfx_texture_get_view(h: number): number;

export function rayzor_gpu_gfx_texture_write(tex_h: number, dev_h: number, data: Uint8Array, bytes_per_row: number, height: number): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly rayzor_gpu_compute_abs: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_add: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_alloc_buffer: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_buffer_dtype: (a: number) => number;
  readonly rayzor_gpu_compute_buffer_numel: (a: number) => number;
  readonly rayzor_gpu_compute_create: () => any;
  readonly rayzor_gpu_compute_destroy: (a: number) => void;
  readonly rayzor_gpu_compute_div: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_dot: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_exp: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_free_buffer: (a: number, b: number) => void;
  readonly rayzor_gpu_compute_gelu: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_is_available: () => number;
  readonly rayzor_gpu_compute_log: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_matmul: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
  readonly rayzor_gpu_compute_max: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_mean: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_min: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_mul: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_neg: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_relu: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_sigmoid: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_silu: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_sqrt: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_sub: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_compute_sum: (a: number, b: number) => number;
  readonly rayzor_gpu_compute_tanh: (a: number, b: number) => number;
  readonly rayzor_gpu_gfx_buffer_create: (a: number, b: number, c: number) => number;
  readonly rayzor_gpu_gfx_buffer_create_with_data: (a: number, b: number, c: number, d: number) => number;
  readonly rayzor_gpu_gfx_buffer_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_buffer_from_bytes: (a: number, b: number, c: number, d: number) => number;
  readonly rayzor_gpu_gfx_buffer_write: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rayzor_gpu_gfx_buffer_write_bytes: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rayzor_gpu_gfx_cmd_begin_pass: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly rayzor_gpu_gfx_cmd_begin_pass_mrt: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly rayzor_gpu_gfx_cmd_create: () => number;
  readonly rayzor_gpu_gfx_cmd_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_cmd_draw: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rayzor_gpu_gfx_cmd_draw_indexed: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly rayzor_gpu_gfx_cmd_end_pass: (a: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_bind_group: (a: number, b: number, c: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_index_buffer: (a: number, b: number, c: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_pipeline: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_scissor: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_vertex_buffer: (a: number, b: number, c: number) => void;
  readonly rayzor_gpu_gfx_cmd_set_viewport: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly rayzor_gpu_gfx_cmd_submit: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_device_create: () => any;
  readonly rayzor_gpu_gfx_device_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_is_available: () => number;
  readonly rayzor_gpu_gfx_pipeline_add_color_target: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_add_layout: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_begin: () => number;
  readonly rayzor_gpu_gfx_pipeline_build: (a: number, b: number) => number;
  readonly rayzor_gpu_gfx_pipeline_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_cull: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_depth_simple: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_format: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_shader: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_topology: (a: number, b: number) => void;
  readonly rayzor_gpu_gfx_pipeline_set_vertex_layout_simple: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly rayzor_gpu_gfx_sampler_create: (a: number, b: number, c: number, d: number) => number;
  readonly rayzor_gpu_gfx_sampler_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_shader_create_hx: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly rayzor_gpu_gfx_shader_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_surface_create: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly rayzor_gpu_gfx_surface_create_canvas: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly rayzor_gpu_gfx_surface_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_surface_get_format: (a: number) => number;
  readonly rayzor_gpu_gfx_surface_get_texture: (a: number) => number;
  readonly rayzor_gpu_gfx_surface_present: (a: number) => void;
  readonly rayzor_gpu_gfx_surface_resize: (a: number, b: number, c: number, d: number) => void;
  readonly rayzor_gpu_gfx_texture_create: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly rayzor_gpu_gfx_texture_destroy: (a: number) => void;
  readonly rayzor_gpu_gfx_texture_get_view: (a: number) => number;
  readonly rayzor_gpu_gfx_texture_write: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wasm_bindgen_335648ada7beb221___JsValue_____: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut__wasm_bindgen_335648ada7beb221___JsValue____Output_______: (a: number, b: number) => void;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wgpu_357224c008ed929d___backend__webgpu__webgpu_sys__gen_GpuUncapturedErrorEvent__GpuUncapturedErrorEvent_____: (a: number, b: number, c: any) => void;
  readonly wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut__wgpu_357224c008ed929d___backend__webgpu__webgpu_sys__gen_GpuUncapturedErrorEvent__GpuUncapturedErrorEvent____Output_______: (a: number, b: number) => void;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke___bool_: (a: number, b: number) => number;
  readonly wasm_bindgen_335648ada7beb221___convert__closures_____invoke___js_sys_fbc68f94bd5fe60e___Function__js_sys_fbc68f94bd5fe60e___Function_____: (a: number, b: number, c: any, d: any) => void;
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
