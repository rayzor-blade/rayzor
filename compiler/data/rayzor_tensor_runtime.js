// Rayzor Tensor Runtime — CPU TypedArrays + GPU offload
// Embedded in JS harness. Uses handle table pattern.
// DType: 0=F32, 1=F16, 2=BF16, 3=I32, 4=I8, 5=U8

const _t = new Map();
let _tN = 1;

function _tMkArr(n, dtype) { return dtype === 3 ? new Int32Array(n) : new Float32Array(n); }
function _tAlloc(data, shape, dtype) {
  const h = _tN++;
  const strides = new Array(shape.length);
  let s = 1;
  for (let i = shape.length - 1; i >= 0; i--) { strides[i] = s; s *= shape[i]; }
  _t.set(h, { data, shape, strides, dtype, numel: data.length });
  return h;
}
function _tGet(h) { return _t.get(h); }
function _tReadShape(mem, shapePtr, ndim) {
  const v = new DataView(mem.buffer);
  const sh = [];
  for (let i = 0; i < ndim; i++) sh.push(v.getInt32(shapePtr + i * 4, true));
  return sh;
}

// Exported as rayzor.rayzor_tensor_* properties
const tensorRuntime = {
  rayzor_tensor_zeros(shapePtr, ndim, dtype) {
    const sh = _tReadShape(memory, shapePtr, ndim);
    return _tAlloc(_tMkArr(sh.reduce((a,b) => a*b, 1), dtype), sh, dtype);
  },
  rayzor_tensor_ones(shapePtr, ndim, dtype) {
    const sh = _tReadShape(memory, shapePtr, ndim);
    const d = _tMkArr(sh.reduce((a,b) => a*b, 1), dtype); d.fill(1);
    return _tAlloc(d, sh, dtype);
  },
  rayzor_tensor_rand(shapePtr, ndim, dtype) {
    const sh = _tReadShape(memory, shapePtr, ndim);
    const n = sh.reduce((a,b) => a*b, 1);
    const d = _tMkArr(n, dtype);
    for (let i = 0; i < n; i++) d[i] = Math.random();
    return _tAlloc(d, sh, dtype);
  },
  rayzor_tensor_full(shapePtr, ndim, val, dtype) {
    const sh = _tReadShape(memory, shapePtr, ndim);
    const d = _tMkArr(sh.reduce((a,b) => a*b, 1), dtype); d.fill(val);
    return _tAlloc(d, sh, dtype);
  },
  rayzor_tensor_from_array(dataPtr, dataLen, dtype) {
    if (!memory) return 0;
    const src = new Float64Array(memory.buffer, dataPtr, dataLen);
    const d = _tMkArr(dataLen, dtype);
    for (let i = 0; i < dataLen; i++) d[i] = src[i];
    return _tAlloc(d, [dataLen], dtype);
  },
  rayzor_tensor_ndim(h) { const t = _tGet(h); return t ? t.shape.length : 0; },
  rayzor_tensor_numel(h) { const t = _tGet(h); return t ? t.numel : 0; },
  rayzor_tensor_dtype(h) { const t = _tGet(h); return t ? t.dtype : 0; },
  rayzor_tensor_shape(h) { return 0; }, // HaxeArray — complex
  rayzor_tensor_shape_ptr(h) { return 0; },
  rayzor_tensor_shape_ndim(h) { return tensorRuntime.rayzor_tensor_ndim(h); },
  rayzor_tensor_get(h, idx) { const t = _tGet(h); return t ? t.data[idx] : 0; },
  rayzor_tensor_set(h, idx, val) { const t = _tGet(h); if (t) t.data[idx] = val; },
  rayzor_tensor_reshape(h, shapePtr, ndim) {
    const t = _tGet(h); if (!t) return 0;
    return _tAlloc(t.data, _tReadShape(memory, shapePtr, ndim), t.dtype);
  },
  rayzor_tensor_transpose(h) {
    const t = _tGet(h); if (!t || t.shape.length !== 2) return 0;
    const [m, n] = t.shape; const d = _tMkArr(m * n, t.dtype);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) d[j * m + i] = t.data[i * n + j];
    return _tAlloc(d, [n, m], t.dtype);
  },
  rayzor_tensor_add(ah, bh) {
    const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0;
    const d = _tMkArr(a.numel, a.dtype);
    for (let i = 0; i < a.numel; i++) d[i] = a.data[i] + b.data[i];
    return _tAlloc(d, [...a.shape], a.dtype);
  },
  rayzor_tensor_sub(ah, bh) {
    const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0;
    const d = _tMkArr(a.numel, a.dtype);
    for (let i = 0; i < a.numel; i++) d[i] = a.data[i] - b.data[i];
    return _tAlloc(d, [...a.shape], a.dtype);
  },
  rayzor_tensor_mul(ah, bh) {
    const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0;
    const d = _tMkArr(a.numel, a.dtype);
    for (let i = 0; i < a.numel; i++) d[i] = a.data[i] * b.data[i];
    return _tAlloc(d, [...a.shape], a.dtype);
  },
  rayzor_tensor_div(ah, bh) {
    const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0;
    const d = _tMkArr(a.numel, a.dtype);
    for (let i = 0; i < a.numel; i++) d[i] = b.data[i] !== 0 ? a.data[i] / b.data[i] : 0;
    return _tAlloc(d, [...a.shape], a.dtype);
  },
  rayzor_tensor_matmul(ah, bh) {
    const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0;
    const m = a.shape[0], k = a.shape[1], n = b.shape[1];
    const d = _tMkArr(m * n, a.dtype);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
      let s = 0; for (let p = 0; p < k; p++) s += a.data[i * k + p] * b.data[p * n + j];
      d[i * n + j] = s;
    }
    return _tAlloc(d, [m, n], a.dtype);
  },
  rayzor_tensor_sqrt(h) { const t = _tGet(h); if (!t) return 0; const d = _tMkArr(t.numel, t.dtype); for (let i = 0; i < t.numel; i++) d[i] = Math.sqrt(t.data[i]); return _tAlloc(d, [...t.shape], t.dtype); },
  rayzor_tensor_exp(h) { const t = _tGet(h); if (!t) return 0; const d = _tMkArr(t.numel, t.dtype); for (let i = 0; i < t.numel; i++) d[i] = Math.exp(t.data[i]); return _tAlloc(d, [...t.shape], t.dtype); },
  rayzor_tensor_log(h) { const t = _tGet(h); if (!t) return 0; const d = _tMkArr(t.numel, t.dtype); for (let i = 0; i < t.numel; i++) d[i] = Math.log(t.data[i]); return _tAlloc(d, [...t.shape], t.dtype); },
  rayzor_tensor_relu(h) { const t = _tGet(h); if (!t) return 0; const d = _tMkArr(t.numel, t.dtype); for (let i = 0; i < t.numel; i++) d[i] = Math.max(0, t.data[i]); return _tAlloc(d, [...t.shape], t.dtype); },
  rayzor_tensor_sum(h) { const t = _tGet(h); if (!t) return 0; let s = 0; for (let i = 0; i < t.numel; i++) s += t.data[i]; return s; },
  rayzor_tensor_mean(h) { const t = _tGet(h); if (!t) return 0; return tensorRuntime.rayzor_tensor_sum(h) / t.numel; },
  rayzor_tensor_dot(ah, bh) { const a = _tGet(ah), b = _tGet(bh); if (!a || !b) return 0; let s = 0; for (let i = 0; i < a.numel; i++) s += a.data[i] * b.data[i]; return s; },
  rayzor_tensor_data(h) { return 0; },
  rayzor_tensor_free(h) { _t.delete(h); },
};
