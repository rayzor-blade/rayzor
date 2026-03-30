let wasm;

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => state.dtor(state.a, state.b));

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeMutClosure(arg0, arg1, dtor, f) {
    const state = { a: arg0, b: arg1, cnt: 1, dtor };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            state.dtor(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

let WASM_VECTOR_LEN = 0;

function wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wasm_bindgen_335648ada7beb221___JsValue_____(arg0, arg1, arg2) {
    wasm.wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wasm_bindgen_335648ada7beb221___JsValue_____(arg0, arg1, arg2);
}

function wasm_bindgen_335648ada7beb221___convert__closures_____invoke______(arg0, arg1) {
    wasm.wasm_bindgen_335648ada7beb221___convert__closures_____invoke______(arg0, arg1);
}

/**
 * @param {string} title
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {number} style
 * @returns {number}
 */
export function rayzor_window_create(title, x, y, w, h, style) {
    const ptr0 = passStringToWasm0(title, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rayzor_window_create(ptr0, len0, x, y, w, h, style);
    return ret;
}

/**
 * @param {string} title
 * @param {number} w
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_create_centered(title, w, h) {
    const ptr0 = passStringToWasm0(title, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.rayzor_window_create_centered(ptr0, len0, w, h);
    return ret;
}

/**
 * @param {number} h
 */
export function rayzor_window_destroy(h) {
    wasm.rayzor_window_destroy(h);
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_button(h, idx) {
    const ret = wasm.rayzor_window_event_button(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_event_count(h) {
    const ret = wasm.rayzor_window_event_count(h);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_height(h, idx) {
    const ret = wasm.rayzor_window_event_height(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_key(h, idx) {
    const ret = wasm.rayzor_window_event_key(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_modifiers(h, idx) {
    const ret = wasm.rayzor_window_event_modifiers(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_scroll_x(h, idx) {
    const ret = wasm.rayzor_window_event_scroll_x(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_scroll_y(h, idx) {
    const ret = wasm.rayzor_window_event_scroll_y(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_type(h, idx) {
    const ret = wasm.rayzor_window_event_type(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_width(h, idx) {
    const ret = wasm.rayzor_window_event_width(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_x(h, idx) {
    const ret = wasm.rayzor_window_event_x(h, idx);
    return ret;
}

/**
 * @param {number} h
 * @param {number} idx
 * @returns {number}
 */
export function rayzor_window_event_y(h, idx) {
    const ret = wasm.rayzor_window_event_y(h, idx);
    return ret;
}

/**
 * @param {number} _h
 * @returns {number}
 */
export function rayzor_window_get_display_handle(_h) {
    const ret = wasm.rayzor_window_get_display_handle(_h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_get_handle(h) {
    const ret = wasm.rayzor_window_get_handle(h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_get_height(h) {
    const ret = wasm.rayzor_window_get_height(h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_get_mouse_x(h) {
    const ret = wasm.rayzor_window_get_mouse_x(h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_get_mouse_y(h) {
    const ret = wasm.rayzor_window_get_mouse_y(h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_get_width(h) {
    const ret = wasm.rayzor_window_get_width(h);
    return ret;
}

/**
 * @param {number} _h
 * @returns {number}
 */
export function rayzor_window_get_x(_h) {
    const ret = wasm.rayzor_window_get_x(_h);
    return ret;
}

/**
 * @param {number} _h
 * @returns {number}
 */
export function rayzor_window_get_y(_h) {
    const ret = wasm.rayzor_window_get_y(_h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_is_focused(h) {
    const ret = wasm.rayzor_window_is_focused(h);
    return ret;
}

/**
 * @param {number} _h
 * @returns {number}
 */
export function rayzor_window_is_fullscreen(_h) {
    const ret = wasm.rayzor_window_is_fullscreen(_h);
    return ret;
}

/**
 * @param {number} h
 * @param {number} key
 * @returns {number}
 */
export function rayzor_window_is_key_down(h, key) {
    const ret = wasm.rayzor_window_is_key_down(h, key);
    return ret;
}

/**
 * @param {number} _h
 * @returns {number}
 */
export function rayzor_window_is_minimized(_h) {
    const ret = wasm.rayzor_window_is_minimized(_h);
    return ret;
}

/**
 * @param {number} h
 * @param {number} button
 * @returns {number}
 */
export function rayzor_window_is_mouse_down(h, button) {
    const ret = wasm.rayzor_window_is_mouse_down(h, button);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_is_visible(h) {
    const ret = wasm.rayzor_window_is_visible(h);
    return ret;
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_poll_events(h) {
    const ret = wasm.rayzor_window_poll_events(h);
    return ret;
}

/**
 * Run a frame-driven render loop using requestAnimationFrame.
 * `callback` is a JS function that returns true to continue, false to stop.
 * On each frame: poll events → call callback → request next frame.
 * @param {number} win_h
 * @param {Function} callback
 */
export function rayzor_window_run_loop(win_h, callback) {
    wasm.rayzor_window_run_loop(win_h, callback);
}

/**
 * @param {number} _h
 * @param {number} _on_top
 */
export function rayzor_window_set_floating(_h, _on_top) {
    wasm.rayzor_window_set_floating(_h, _on_top);
}

/**
 * @param {number} h
 * @param {number} fs
 */
export function rayzor_window_set_fullscreen(h, fs) {
    wasm.rayzor_window_set_fullscreen(h, fs);
}

/**
 * @param {number} _h
 * @param {number} _w
 * @param {number} _ht
 */
export function rayzor_window_set_max_size(_h, _w, _ht) {
    wasm.rayzor_window_set_max_size(_h, _w, _ht);
}

/**
 * @param {number} _h
 * @param {number} _w
 * @param {number} _ht
 */
export function rayzor_window_set_min_size(_h, _w, _ht) {
    wasm.rayzor_window_set_min_size(_h, _w, _ht);
}

/**
 * @param {number} h
 * @param {number} opacity
 */
export function rayzor_window_set_opacity(h, opacity) {
    wasm.rayzor_window_set_opacity(h, opacity);
}

/**
 * @param {number} _h
 * @param {number} _x
 * @param {number} _y
 */
export function rayzor_window_set_position(_h, _x, _y) {
    wasm.rayzor_window_set_position(_h, _x, _y);
}

/**
 * @param {number} h
 * @param {number} w
 * @param {number} ht_val
 */
export function rayzor_window_set_size(h, w, ht_val) {
    wasm.rayzor_window_set_size(h, w, ht_val);
}

/**
 * @param {number} _h
 * @param {string} title
 */
export function rayzor_window_set_title(_h, title) {
    const ptr0 = passStringToWasm0(title, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    wasm.rayzor_window_set_title(_h, ptr0, len0);
}

/**
 * @param {number} h
 * @param {number} vis
 */
export function rayzor_window_set_visible(h, vis) {
    wasm.rayzor_window_set_visible(h, vis);
}

/**
 * @param {number} h
 * @returns {number}
 */
export function rayzor_window_was_resized(h) {
    const ret = wasm.rayzor_window_was_resized(h);
    return ret;
}

const EXPECTED_RESPONSE_TYPES = new Set(['basic', 'cors', 'default']);

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && EXPECTED_RESPONSE_TYPES.has(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else {
                    throw e;
                }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }
}

function __wbg_get_imports() {
    const imports = {};
    imports.wbg = {};
    imports.wbg.__wbg___wbindgen_boolean_get_dea25b33882b895b = function(arg0) {
        const v = arg0;
        const ret = typeof(v) === 'boolean' ? v : undefined;
        return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
    };
    imports.wbg.__wbg___wbindgen_debug_string_adfb662ae34724b6 = function(arg0, arg1) {
        const ret = debugString(arg1);
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg___wbindgen_is_undefined_f6b95eab589e0269 = function(arg0) {
        const ret = arg0 === undefined;
        return ret;
    };
    imports.wbg.__wbg___wbindgen_throw_dd24417ed36fc46e = function(arg0, arg1) {
        throw new Error(getStringFromWasm0(arg0, arg1));
    };
    imports.wbg.__wbg__wbg_cb_unref_87dfb5aaa0cbcea7 = function(arg0) {
        arg0._wbg_cb_unref();
    };
    imports.wbg.__wbg_addEventListener_6a82629b3d430a48 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        arg0.addEventListener(getStringFromWasm0(arg1, arg2), arg3);
    }, arguments) };
    imports.wbg.__wbg_addEventListener_82cddc614107eb45 = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.addEventListener(getStringFromWasm0(arg1, arg2), arg3, arg4);
    }, arguments) };
    imports.wbg.__wbg_altKey_56d1d642f3a28c92 = function(arg0) {
        const ret = arg0.altKey;
        return ret;
    };
    imports.wbg.__wbg_appendChild_7465eba84213c75f = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.appendChild(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_body_544738f8b03aef13 = function(arg0) {
        const ret = arg0.body;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_button_a54acd25bab5d442 = function(arg0) {
        const ret = arg0.button;
        return ret;
    };
    imports.wbg.__wbg_call_abb4ff46ce38be40 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.call(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_clientX_c17906c33ea43025 = function(arg0) {
        const ret = arg0.clientX;
        return ret;
    };
    imports.wbg.__wbg_clientY_70eb66d231a332a3 = function(arg0) {
        const ret = arg0.clientY;
        return ret;
    };
    imports.wbg.__wbg_code_b3ddfa90f724c486 = function(arg0, arg1) {
        const ret = arg1.code;
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    };
    imports.wbg.__wbg_createElement_da4ed2b219560fc6 = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.createElement(getStringFromWasm0(arg1, arg2));
        return ret;
    }, arguments) };
    imports.wbg.__wbg_ctrlKey_487597b9069da036 = function(arg0) {
        const ret = arg0.ctrlKey;
        return ret;
    };
    imports.wbg.__wbg_deltaX_41f7678c94b10355 = function(arg0) {
        const ret = arg0.deltaX;
        return ret;
    };
    imports.wbg.__wbg_deltaY_3f10fd796fae2a0f = function(arg0) {
        const ret = arg0.deltaY;
        return ret;
    };
    imports.wbg.__wbg_document_5b745e82ba551ca5 = function(arg0) {
        const ret = arg0.document;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_focus_220a53e22147dc0f = function() { return handleError(function (arg0) {
        arg0.focus();
    }, arguments) };
    imports.wbg.__wbg_fullscreenElement_e2e939644adf50e1 = function(arg0) {
        const ret = arg0.fullscreenElement;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_getBoundingClientRect_25e44a78507968b0 = function(arg0) {
        const ret = arg0.getBoundingClientRect();
        return ret;
    };
    imports.wbg.__wbg_getPropertyValue_dcded91357966805 = function() { return handleError(function (arg0, arg1, arg2, arg3) {
        const ret = arg1.getPropertyValue(getStringFromWasm0(arg2, arg3));
        const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
        getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
    }, arguments) };
    imports.wbg.__wbg_hidden_63c9db3ea5c1e10a = function(arg0) {
        const ret = arg0.hidden;
        return ret;
    };
    imports.wbg.__wbg_instanceof_HtmlCanvasElement_c4251b1b6a15edcc = function(arg0) {
        let result;
        try {
            result = arg0 instanceof HTMLCanvasElement;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_KeyboardEvent_1d6c7f5fcec88195 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof KeyboardEvent;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_MouseEvent_4044e738005b8891 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof MouseEvent;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_WheelEvent_126f1ae9bf322f38 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof WheelEvent;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_instanceof_Window_b5cf7783caa68180 = function(arg0) {
        let result;
        try {
            result = arg0 instanceof Window;
        } catch (_) {
            result = false;
        }
        const ret = result;
        return ret;
    };
    imports.wbg.__wbg_left_d52bfa3824286825 = function(arg0) {
        const ret = arg0.left;
        return ret;
    };
    imports.wbg.__wbg_metaKey_0572b1cbcb5b272b = function(arg0) {
        const ret = arg0.metaKey;
        return ret;
    };
    imports.wbg.__wbg_new_1ba21ce319a06297 = function() {
        const ret = new Object();
        return ret;
    };
    imports.wbg.__wbg_new_no_args_cb138f77cf6151ee = function(arg0, arg1) {
        const ret = new Function(getStringFromWasm0(arg0, arg1));
        return ret;
    };
    imports.wbg.__wbg_preventDefault_e97663aeeb9709d3 = function(arg0) {
        arg0.preventDefault();
    };
    imports.wbg.__wbg_querySelector_15a92ce6bed6157d = function() { return handleError(function (arg0, arg1, arg2) {
        const ret = arg0.querySelector(getStringFromWasm0(arg1, arg2));
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    }, arguments) };
    imports.wbg.__wbg_requestAnimationFrame_994dc4ebde22b8d9 = function() { return handleError(function (arg0, arg1) {
        const ret = arg0.requestAnimationFrame(arg1);
        return ret;
    }, arguments) };
    imports.wbg.__wbg_requestFullscreen_0d9f7148d4658c31 = function() { return handleError(function (arg0) {
        arg0.requestFullscreen();
    }, arguments) };
    imports.wbg.__wbg_setProperty_f27b2c05323daf8a = function() { return handleError(function (arg0, arg1, arg2, arg3, arg4) {
        arg0.setProperty(getStringFromWasm0(arg1, arg2), getStringFromWasm0(arg3, arg4));
    }, arguments) };
    imports.wbg.__wbg_set_height_6f8f8ef4cb40e496 = function(arg0, arg1) {
        arg0.height = arg1 >>> 0;
    };
    imports.wbg.__wbg_set_id_702da6e1bcec3b45 = function(arg0, arg1, arg2) {
        arg0.id = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_passive_a3aa35eb7292414e = function(arg0, arg1) {
        arg0.passive = arg1 !== 0;
    };
    imports.wbg.__wbg_set_tabIndex_10b13c5f00904478 = function(arg0, arg1) {
        arg0.tabIndex = arg1;
    };
    imports.wbg.__wbg_set_title_68ffc586125a93b4 = function(arg0, arg1, arg2) {
        arg0.title = getStringFromWasm0(arg1, arg2);
    };
    imports.wbg.__wbg_set_width_7ff7a22c6e9f423e = function(arg0, arg1) {
        arg0.width = arg1 >>> 0;
    };
    imports.wbg.__wbg_shiftKey_d2640abcfa98acec = function(arg0) {
        const ret = arg0.shiftKey;
        return ret;
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_769e6b65d6557335 = function() {
        const ret = typeof global === 'undefined' ? null : global;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_GLOBAL_THIS_60cf02db4de8e1c1 = function() {
        const ret = typeof globalThis === 'undefined' ? null : globalThis;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_SELF_08f5a74c69739274 = function() {
        const ret = typeof self === 'undefined' ? null : self;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_static_accessor_WINDOW_a8924b26aa92d024 = function() {
        const ret = typeof window === 'undefined' ? null : window;
        return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
    };
    imports.wbg.__wbg_style_521a717da50e53c6 = function(arg0) {
        const ret = arg0.style;
        return ret;
    };
    imports.wbg.__wbg_top_7d5b82a2c5d7f13f = function(arg0) {
        const ret = arg0.top;
        return ret;
    };
    imports.wbg.__wbindgen_cast_a6a21deee28bc5cc = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 42, function: Function { arguments: [Externref], shim_idx: 11, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, wasm.wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut__wasm_bindgen_335648ada7beb221___JsValue____Output_______, wasm_bindgen_335648ada7beb221___convert__closures_____invoke___wasm_bindgen_335648ada7beb221___JsValue_____);
        return ret;
    };
    imports.wbg.__wbindgen_cast_b4d5e2bb57fb3a56 = function(arg0, arg1) {
        // Cast intrinsic for `Closure(Closure { dtor_idx: 43, function: Function { arguments: [], shim_idx: 10, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
        const ret = makeMutClosure(arg0, arg1, wasm.wasm_bindgen_335648ada7beb221___closure__destroy___dyn_core_e0615fd90a40850c___ops__function__FnMut_____Output_______, wasm_bindgen_335648ada7beb221___convert__closures_____invoke______);
        return ret;
    };
    imports.wbg.__wbindgen_init_externref_table = function() {
        const table = wasm.__wbindgen_externrefs;
        const offset = table.grow(4);
        table.set(0, undefined);
        table.set(offset + 0, undefined);
        table.set(offset + 1, null);
        table.set(offset + 2, true);
        table.set(offset + 3, false);
    };

    return imports;
}

function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    __wbg_init.__wbindgen_wasm_module = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;


    wasm.__wbindgen_start();
    return wasm;
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (typeof module !== 'undefined') {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (typeof module_or_path !== 'undefined') {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (typeof module_or_path === 'undefined') {
        module_or_path = new URL('rayzor_window_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync };
export default __wbg_init;
