// Rayzor GPU Host — Browser WebGPU implementation of rayzor:gpu interface.
//
// Provides the JS-side implementation of all GPU compute + graphics functions
// that the WASM module imports via @:jsImport("rayzor-gpu").
//
// The host is initialized with a reference to WASM memory so it can read
// HaxeString pointers and buffer data from linear memory.

export function createGpuHost(getMemory) {
  // Handle table: i32 handles → JS GPU objects
  let nextHandle = 1;
  const handles = new Map();
  const alloc = (obj) => { const h = nextHandle++; handles.set(h, obj); return h; };
  const get = (h) => handles.get(h) ?? null;
  const free = (h) => handles.delete(h);

  let device = null;
  let queue = null;
  let adapter = null;

  // Read a HaxeString from WASM memory: { ptr: u32, len: u32, cap: u32 }
  function readHaxeString(hsPtr) {
    if (!hsPtr) return '';
    const mem = getMemory();
    if (!mem) return '';
    const view = new DataView(mem.buffer);
    const dataPtr = view.getUint32(hsPtr, true);
    const len = view.getUint32(hsPtr + 4, true);
    if (!dataPtr || !len) return '';
    return new TextDecoder().decode(new Uint8Array(mem.buffer, dataPtr, len));
  }

  // Read raw bytes from WASM memory
  function readBytes(ptr, size) {
    const mem = getMemory();
    if (!mem || !ptr) return null;
    return new Uint8Array(mem.buffer, ptr, size);
  }

  // --- Texture format mapping (matches GraphicsTypes.hx) ---
  const FORMATS = {
    0: 'bgra8unorm', 1: 'bgra8unorm', 2: 'rgba8unorm',
    3: 'depth24plus-stencil8', 4: 'depth32float',
    5: 'rgba16float', 6: 'rgba32float',
    7: 'bgra8unorm-srgb', 8: 'rgba8unorm-srgb',
  };
  const TOPOLOGY = { 0: 'triangle-list', 1: 'triangle-strip', 2: 'line-list', 3: 'line-strip', 4: 'point-list' };
  const CULL = { 0: 'none', 1: 'front', 2: 'back' };
  const INDEX_FMT = { 0: 'uint16', 1: 'uint32' };

  // --- Compute shader templates ---
  function binaryShader(op) {
    return `@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i < arrayLength(&a)) { out[i] = a[i] ${op} b[i]; }
}`;
  }
  function unaryShader(expr) {
    return `@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i < arrayLength(&a)) { out[i] = ${expr}; }
}`;
  }
  const REDUCE_SHADER = `@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x == 0u) {
    var s: f32 = 0.0;
    for (var i = 0u; i < arrayLength(&a); i++) { s += a[i]; }
    out[0] = s;
  }
}`;

  // Cache compiled compute pipelines
  const pipelineCache = new Map();

  function getComputePipeline(wgsl) {
    if (pipelineCache.has(wgsl)) return pipelineCache.get(wgsl);
    const module = device.createShaderModule({ code: wgsl });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    pipelineCache.set(wgsl, pipeline);
    return pipeline;
  }

  function runBinaryOp(op, aHandle, bHandle) {
    const a = get(aHandle), b = get(bHandle);
    if (!a?.buf || !b?.buf || !device) return 0;
    const n = a.numel;
    const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const pipeline = getComputePipeline(binaryShader(op));
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: a.buf } },
      { binding: 1, resource: { buffer: b.buf } },
      { binding: 2, resource: { buffer: outBuf } },
    ]});
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
    queue.submit([enc.finish()]);
    return alloc({ type: 'buffer', buf: outBuf, numel: n, dtype: a.dtype });
  }

  function runUnaryOp(expr, aHandle) {
    const a = get(aHandle);
    if (!a?.buf || !device) return 0;
    const n = a.numel;
    const outBuf = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const pipeline = getComputePipeline(unaryShader(expr));
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: a.buf } },
      { binding: 1, resource: { buffer: outBuf } },
    ]});
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
    queue.submit([enc.finish()]);
    return alloc({ type: 'buffer', buf: outBuf, numel: n, dtype: a.dtype });
  }

  async function reduceSum(bufHandle) {
    const b = get(bufHandle);
    if (!b?.buf || !device) return 0;
    const n = b.numel;
    const outBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipeline = getComputePipeline(REDUCE_SHADER);
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: b.buf } },
      { binding: 1, resource: { buffer: outBuf } },
    ]});
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4);
    queue.submit([enc.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange())[0];
    readBuf.unmap();
    outBuf.destroy();
    readBuf.destroy();
    return result;
  }

  return {
    // ---- Compute ----
    'create-device': () => device ? alloc({ type: 'compute' }) : 0,
    'is-available': () => !!navigator.gpu,
    'destroy-device': (dev) => free(dev),

    'alloc-buffer': (_dev, numel, dtype) => {
      if (!device) return 0;
      const size = Math.max(numel * 4, 4);
      const buf = device.createBuffer({
        size: Math.ceil(size / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      return alloc({ type: 'buffer', buf, numel, dtype });
    },
    'free-buffer': (_dev, buf) => { const b = get(buf); if (b?.buf) b.buf.destroy(); free(buf); },
    'buffer-numel': (buf) => get(buf)?.numel ?? 0,
    'buffer-dtype': (buf) => get(buf)?.dtype ?? 0,

    'add': (_d, a, b) => runBinaryOp('+', a, b),
    'sub': (_d, a, b) => runBinaryOp('-', a, b),
    'mul': (_d, a, b) => runBinaryOp('*', a, b),
    'div': (_d, a, b) => runBinaryOp('/', a, b),
    'neg': (_d, a) => runUnaryOp('-a[i]', a),
    'abs': (_d, a) => runUnaryOp('abs(a[i])', a),
    'sqrt': (_d, a) => runUnaryOp('sqrt(a[i])', a),
    'exp': (_d, a) => runUnaryOp('exp(a[i])', a),
    'log': (_d, a) => runUnaryOp('log(a[i])', a),
    'relu': (_d, a) => runUnaryOp('max(a[i], 0.0)', a),
    'sigmoid': (_d, a) => runUnaryOp('1.0 / (1.0 + exp(-a[i]))', a),
    'gpu-tanh': (_d, a) => runUnaryOp('tanh(a[i])', a),
    'gelu': (_d, a) => runUnaryOp('a[i] * 0.5 * (1.0 + tanh(0.7978845608 * (a[i] + 0.044715 * a[i] * a[i] * a[i])))', a),
    'silu': (_d, a) => runUnaryOp('a[i] / (1.0 + exp(-a[i]))', a),

    // Reductions (synchronous wrappers — may block briefly for GPU readback)
    'sum': (_d, buf) => { let r = 0; reduceSum(buf).then(v => r = v); return r; },
    'mean': (_d, buf) => { const b = get(buf); let r = 0; reduceSum(buf).then(v => r = v / (b?.numel || 1)); return r; },
    'reduce-max': (_d, buf) => 0.0, // TODO: max reduction shader
    'reduce-min': (_d, buf) => 0.0, // TODO: min reduction shader
    'dot': (_d, a, b) => 0.0, // TODO: dot product shader
    'matmul': (_d, a, b, m, k, n) => 0, // TODO: matmul shader
    'batch-matmul': (_d, a, b, batch, m, k, n) => 0,

    // ---- Graphics ----
    'gfx-create-device': () => device ? alloc({ type: 'gfx' }) : 0,
    'gfx-destroy-device': (dev) => free(dev),
    'gfx-is-available': () => !!navigator.gpu ? 1 : 0,

    'create-surface-canvas': (_dev, canvasIdPtr, width, height) => {
      let canvas = document.querySelector('canvas');
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'rayzor-canvas';
        canvas.style.cssText = 'display:block;margin:0 auto;background:#111;';
        document.body.appendChild(canvas);
      }
      canvas.width = width || 800;
      canvas.height = height || 600;
      if (!device) return 0;
      const ctx = canvas.getContext('webgpu');
      if (!ctx) return 0;
      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format, alphaMode: 'premultiplied' });
      return alloc({ type: 'surface', canvas, ctx, format });
    },
    'create-surface': (_dev, _wh, _dh, w, h) => {
      // Browser: delegate to canvas surface
      return createGpuHost(getMemory)['create-surface-canvas'](_dev, 0, w, h);
    },
    'surface-get-texture': (surf) => {
      const s = get(surf);
      if (!s?.ctx) return 0;
      try {
        const tex = s.ctx.getCurrentTexture();
        return alloc({ type: 'textureView', view: tex.createView(), tex });
      } catch { return 0; }
    },
    'surface-present': () => { /* browser auto-presents */ },
    'surface-resize': (surf, _dev, w, h) => {
      const s = get(surf);
      if (!s) return;
      s.canvas.width = w; s.canvas.height = h;
      if (device) s.ctx.configure({ device, format: s.format, alphaMode: 'premultiplied' });
    },
    'surface-get-format': (surf) => {
      const s = get(surf);
      const fmt = s?.format ?? 'bgra8unorm';
      return fmt.includes('srgb') ? 7 : 1;
    },
    'surface-destroy': (surf) => free(surf),

    // Shaders
    'create-shader': (_dev, wgslPtr, vsEntryPtr, fsEntryPtr) => {
      if (!device) return 0;
      const wgsl = readHaxeString(wgslPtr);
      const vsEntry = readHaxeString(vsEntryPtr) || 'vs_main';
      const fsEntry = readHaxeString(fsEntryPtr) || 'fs_main';
      if (!wgsl) return 0;
      try {
        const module = device.createShaderModule({ code: wgsl });
        return alloc({ type: 'shader', module, vsEntry, fsEntry });
      } catch (e) { console.error('[rayzor:gpu] Shader error:', e); return 0; }
    },
    'destroy-shader': (s) => free(s),

    // Buffers
    'create-buffer': (_dev, size, usage) => {
      if (!device) return 0;
      const buf = device.createBuffer({ size: Math.max(size, 4), usage, mappedAtCreation: false });
      return alloc({ type: 'gfxBuffer', buf, size });
    },
    'create-buffer-with-data': (_dev, dataPtr, size, usage) => {
      if (!device) return 0;
      const buf = device.createBuffer({ size: Math.max(size, 4), usage, mappedAtCreation: true });
      const data = readBytes(dataPtr, size);
      if (data) new Uint8Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return alloc({ type: 'gfxBuffer', buf, size });
    },
    'buffer-write': (bufH, _dev, offset, dataPtr, size) => {
      const b = get(bufH);
      if (!b?.buf || !device) return;
      const data = readBytes(dataPtr, size);
      if (data) queue.writeBuffer(b.buf, offset, data);
    },
    'buffer-destroy': (bufH) => { const b = get(bufH); if (b?.buf) b.buf.destroy(); free(bufH); },

    // Textures
    'create-texture': (_dev, w, h, fmt, usage) => {
      if (!device) return 0;
      const tex = device.createTexture({
        size: [w, h], format: FORMATS[fmt] || 'rgba8unorm', usage,
      });
      return alloc({ type: 'texture', tex, w, h });
    },
    'texture-write': (texH, _dev, dataPtr, bytesPerRow, height) => {
      const t = get(texH);
      if (!t?.tex || !device) return;
      const data = readBytes(dataPtr, bytesPerRow * height);
      if (data) queue.writeTexture({ texture: t.tex }, data, { bytesPerRow }, [t.w, height]);
    },
    'texture-get-view': (texH) => {
      const t = get(texH);
      if (!t?.tex) return 0;
      return alloc({ type: 'textureView', view: t.tex.createView() });
    },
    'texture-destroy': (texH) => { const t = get(texH); if (t?.tex) t.tex.destroy(); free(texH); },
    'texture-read-pixels': () => 0, // TODO: readback

    // Sampler
    'sampler-create': (_dev, minFilter, magFilter, addressMode) => {
      if (!device) return 0;
      const filterMap = { 0: 'nearest', 1: 'linear' };
      const addrMap = { 0: 'clamp-to-edge', 1: 'repeat', 2: 'mirror-repeat' };
      const sampler = device.createSampler({
        minFilter: filterMap[minFilter] || 'nearest',
        magFilter: filterMap[magFilter] || 'nearest',
        addressModeU: addrMap[addressMode] || 'clamp-to-edge',
        addressModeV: addrMap[addressMode] || 'clamp-to-edge',
      });
      return alloc({ type: 'sampler', sampler });
    },
    'sampler-destroy': (s) => free(s),

    // Pipeline builder
    'pipeline-begin': () => alloc({
      type: 'pipelineBuilder', shader: null, format: 'bgra8unorm',
      topology: 'triangle-list', cullMode: 'none', colorTargets: [],
    }),
    'pipeline-set-shader': (pipe, shaderH) => { const p = get(pipe), s = get(shaderH); if (p && s) p.shader = s; },
    'pipeline-set-format': (pipe, fmt) => { const p = get(pipe); if (p) p.format = FORMATS[fmt] || 'bgra8unorm'; },
    'pipeline-set-topology': (pipe, topo) => { const p = get(pipe); if (p) p.topology = TOPOLOGY[topo] || 'triangle-list'; },
    'pipeline-set-cull': (pipe, cull) => { const p = get(pipe); if (p) p.cullMode = CULL[cull] || 'none'; },
    'pipeline-add-color-target': (pipe, fmt) => { const p = get(pipe); if (p) p.colorTargets.push(FORMATS[fmt] || 'bgra8unorm'); },
    'pipeline-build': (pipe, _dev) => {
      const p = get(pipe);
      if (!p?.shader?.module || !device) return 0;
      const targets = p.colorTargets.length > 0
        ? p.colorTargets.map(f => ({ format: f }))
        : [{ format: p.format }];
      try {
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module: p.shader.module, entryPoint: p.shader.vsEntry },
          fragment: { module: p.shader.module, entryPoint: p.shader.fsEntry, targets },
          primitive: { topology: p.topology, cullMode: p.cullMode },
        });
        return alloc({ type: 'pipeline', pipeline });
      } catch (e) { console.error('[rayzor:gpu] Pipeline error:', e); return 0; }
    },
    'pipeline-destroy': (pipe) => free(pipe),

    // Command encoder
    'cmd-create': () => {
      if (!device) return 0;
      const encoder = device.createCommandEncoder();
      return alloc({ type: 'cmd', encoder, currentPass: null });
    },
    'cmd-begin-pass': (cmdH, colorViewH, loadOp, r, g, b, a, depthViewH) => {
      const cmd = get(cmdH), cv = get(colorViewH), dv = get(depthViewH);
      if (!cmd?.encoder || !cv?.view) return;
      const colorAttachment = {
        view: cv.view,
        loadOp: loadOp === 0 ? 'clear' : 'load',
        storeOp: 'store',
        clearValue: { r, g, b, a },
      };
      const desc = { colorAttachments: [colorAttachment] };
      if (dv?.view) {
        desc.depthStencilAttachment = {
          view: dv.view, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0,
        };
      }
      cmd.currentPass = cmd.encoder.beginRenderPass(desc);
    },
    'cmd-end-pass': (cmdH) => { const cmd = get(cmdH); if (cmd?.currentPass) { cmd.currentPass.end(); cmd.currentPass = null; } },
    'cmd-submit': (cmdH, _dev) => {
      const cmd = get(cmdH);
      if (!cmd?.encoder || !device) return;
      queue.submit([cmd.encoder.finish()]);
      // Allocate new encoder for potential reuse
      cmd.encoder = device.createCommandEncoder();
    },
    'cmd-set-pipeline': (cmdH, pipeH) => {
      const cmd = get(cmdH), p = get(pipeH);
      if (cmd?.currentPass && p?.pipeline) cmd.currentPass.setPipeline(p.pipeline);
    },
    'cmd-draw': (cmdH, vertexCount, instanceCount, firstVertex, firstInstance) => {
      const cmd = get(cmdH);
      if (cmd?.currentPass) cmd.currentPass.draw(vertexCount, instanceCount, firstVertex, firstInstance);
    },
    'cmd-draw-indexed': (cmdH, indexCount, instanceCount, firstIndex, baseVertex, firstInstance) => {
      const cmd = get(cmdH);
      if (cmd?.currentPass) cmd.currentPass.drawIndexed(indexCount, instanceCount, firstIndex, baseVertex, firstInstance);
    },
    'cmd-set-vertex-buffer': (cmdH, slot, bufH) => {
      const cmd = get(cmdH), b = get(bufH);
      if (cmd?.currentPass && b?.buf) cmd.currentPass.setVertexBuffer(slot, b.buf);
    },
    'cmd-set-index-buffer': (cmdH, bufH, fmt) => {
      const cmd = get(cmdH), b = get(bufH);
      if (cmd?.currentPass && b?.buf) cmd.currentPass.setIndexBuffer(b.buf, INDEX_FMT[fmt] || 'uint16');
    },
    'cmd-set-bind-group': (cmdH, groupIdx, bgH) => {
      const cmd = get(cmdH), bg = get(bgH);
      if (cmd?.currentPass && bg?.bindGroup) cmd.currentPass.setBindGroup(groupIdx, bg.bindGroup);
    },
    'cmd-set-viewport': (cmdH, x, y, w, h, minDepth, maxDepth) => {
      const cmd = get(cmdH);
      if (cmd?.currentPass) cmd.currentPass.setViewport(x, y, w, h, minDepth, maxDepth);
    },
    'cmd-set-scissor': (cmdH, x, y, w, h) => {
      const cmd = get(cmdH);
      if (cmd?.currentPass) cmd.currentPass.setScissorRect(x, y, w, h);
    },

    // Convenience rendering
    'render-triangles': (_dev, colorViewH, pipeH, vertexCount, r, g, b, a) => {
      const cv = get(colorViewH), p = get(pipeH);
      if (!cv?.view || !p?.pipeline || !device) return;
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: cv.view, loadOp: 'clear', storeOp: 'store',
          clearValue: { r, g, b, a },
        }],
      });
      pass.setPipeline(p.pipeline);
      pass.draw(vertexCount);
      pass.end();
      queue.submit([enc.finish()]);
    },

    // Initialization — must be called before WASM start
    async init() {
      if (!navigator.gpu) return false;
      try {
        adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) return false;
        device = await adapter.requestDevice();
        queue = device.queue;
        return true;
      } catch (e) {
        console.warn('[rayzor:gpu] WebGPU init failed:', e);
        return false;
      }
    },
  };
}

export default createGpuHost;
