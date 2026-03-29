// Rayzor GPU Host — Browser implementation of rayzor:gpu WIT interface.
//
// Provides WebGPU (primary) with WebGL2 fallback for GPU compute + graphics.
// This file is included in the generated JS harness when GPU rpkg is detected.

const GPU_HOST = (() => {
  // Handle table: maps i32 handles → JS objects
  let nextHandle = 1;
  const handles = new Map();

  function allocHandle(obj) {
    const h = nextHandle++;
    handles.set(h, obj);
    return h;
  }
  function getHandle(h) { return handles.get(h) ?? null; }
  function freeHandle(h) { handles.delete(h); }

  // WebGPU device (lazily initialized)
  let gpuDevice = null;
  let gpuQueue = null;
  let gpuAdapter = null;

  async function ensureDevice() {
    if (gpuDevice) return true;
    if (!navigator.gpu) return false;
    try {
      gpuAdapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!gpuAdapter) return false;
      gpuDevice = await gpuAdapter.requestDevice();
      gpuQueue = gpuDevice.queue;
      return true;
    } catch (e) {
      console.warn('[rayzor:gpu] WebGPU init failed:', e);
      return false;
    }
  }

  return {
    // ---- compute interface ----
    'create-device': () => {
      // Synchronous check — device must be pre-initialized via init()
      return gpuDevice ? allocHandle({ type: 'compute', device: gpuDevice }) : 0;
    },
    'is-available': () => !!navigator.gpu,
    'destroy-device': (dev) => freeHandle(dev),
    'alloc-buffer': (dev, numel, dtype) => {
      if (!gpuDevice) return 0;
      const bytesPerElem = dtype === 2 ? 8 : 4; // F64=8, else 4
      const size = Math.max(numel * bytesPerElem, 4);
      const buf = gpuDevice.createBuffer({
        size: Math.ceil(size / 4) * 4, // align to 4
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      return allocHandle({ type: 'buffer', buf, numel, dtype });
    },
    'free-buffer': (dev, buf) => {
      const b = getHandle(buf);
      if (b?.buf) b.buf.destroy();
      freeHandle(buf);
    },
    'buffer-numel': (buf) => getHandle(buf)?.numel ?? 0,

    // Binary ops (stubs — full compute shader pipeline needed for real impl)
    'add': (dev, a, b) => 0,
    'sub': (dev, a, b) => 0,
    'mul': (dev, a, b) => 0,
    'div': (dev, a, b) => 0,
    'neg': (dev, a) => 0,
    'abs': (dev, a) => 0,
    'sqrt': (dev, a) => 0,
    'exp': (dev, a) => 0,
    'log': (dev, a) => 0,
    'relu': (dev, a) => 0,
    'sigmoid': (dev, a) => 0,
    'gpu-tanh': (dev, a) => 0,
    'sum': (dev, buf) => 0.0,
    'mean': (dev, buf) => 0.0,
    'reduce-max': (dev, buf) => 0.0,
    'reduce-min': (dev, buf) => 0.0,
    'dot': (dev, a, b) => 0.0,
    'matmul': (dev, a, b, m, k, n) => 0,

    // ---- graphics interface ----
    'gfx-create-device': () => {
      return gpuDevice ? allocHandle({ type: 'gfx', device: gpuDevice }) : 0;
    },
    'gfx-destroy-device': (dev) => freeHandle(dev),
    'gfx-is-available': () => !!navigator.gpu,

    'create-surface-canvas': (dev, canvasIdPtr, width, height) => {
      // Find or create canvas element
      let canvas = document.querySelector('canvas');
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'rayzor-canvas';
        document.body.appendChild(canvas);
      }
      canvas.width = width || 800;
      canvas.height = height || 600;

      if (!gpuDevice) return 0;
      const ctx = canvas.getContext('webgpu');
      if (!ctx) {
        // WebGL2 fallback
        const gl = canvas.getContext('webgl2');
        return gl ? allocHandle({ type: 'surface', canvas, gl, isWebGL: true }) : 0;
      }

      const format = navigator.gpu.getPreferredCanvasFormat();
      ctx.configure({ device: gpuDevice, format, alphaMode: 'premultiplied' });
      return allocHandle({ type: 'surface', canvas, ctx, format, isWebGL: false });
    },

    'create-surface': (dev, windowHandle, displayHandle, width, height) => {
      // On browser, surface-create falls back to canvas creation
      return GPU_HOST['create-surface-canvas'](dev, 0, width, height);
    },

    'surface-get-texture': (surf) => {
      const s = getHandle(surf);
      if (!s || s.isWebGL) return 0;
      try {
        const tex = s.ctx.getCurrentTexture();
        const view = tex.createView();
        return allocHandle({ type: 'textureView', view, tex });
      } catch (e) { return 0; }
    },
    'surface-present': (surf) => { /* browser auto-presents at end of frame */ },
    'surface-resize': (surf, dev, width, height) => {
      const s = getHandle(surf);
      if (!s) return;
      s.canvas.width = width;
      s.canvas.height = height;
      if (!s.isWebGL && gpuDevice) {
        s.ctx.configure({ device: gpuDevice, format: s.format, alphaMode: 'premultiplied' });
      }
    },
    'surface-get-format': (surf) => {
      const s = getHandle(surf);
      return s?.format === 'bgra8unorm-srgb' ? 7 : 1; // BGRA8UnormSrgb=7, BGRA8Unorm=1
    },
    'surface-destroy': (surf) => freeHandle(surf),

    'create-shader': (dev, wgslSource, vsEntry, fsEntry) => {
      if (!gpuDevice) return 0;
      // wgslSource is a WASM memory pointer — need to read string from memory
      // This will be wired up when the host harness passes memory access
      return 0; // TODO: read WGSL from WASM memory, create shader module
    },
    'destroy-shader': (shader) => freeHandle(shader),

    'create-buffer': (dev, size, usageFlags) => {
      if (!gpuDevice) return 0;
      const buf = gpuDevice.createBuffer({ size: Math.max(size, 4), usage: usageFlags });
      return allocHandle({ type: 'gfxBuffer', buf });
    },
    'create-buffer-with-data': (dev, dataPtr, size, usageFlags) => {
      return GPU_HOST['create-buffer'](dev, size, usageFlags);
    },
    'buffer-write': (buf, dev, offset, dataPtr, size) => { /* TODO: write from WASM memory */ },
    'buffer-destroy': (buf) => {
      const b = getHandle(buf);
      if (b?.buf) b.buf.destroy();
      freeHandle(buf);
    },

    'create-texture': (dev, width, height, format, usageFlags) => {
      if (!gpuDevice) return 0;
      const tex = gpuDevice.createTexture({
        size: [width, height],
        format: 'rgba8unorm',
        usage: usageFlags,
      });
      return allocHandle({ type: 'texture', tex });
    },
    'texture-write': () => {},
    'texture-get-view': (tex) => {
      const t = getHandle(tex);
      if (!t?.tex) return 0;
      return allocHandle({ type: 'textureView', view: t.tex.createView() });
    },
    'texture-destroy': (tex) => {
      const t = getHandle(tex);
      if (t?.tex) t.tex.destroy();
      freeHandle(tex);
    },

    'pipeline-begin': () => allocHandle({ type: 'pipelineBuilder', config: {} }),
    'pipeline-set-shader': (pipe, shader) => {},
    'pipeline-set-format': (pipe, format) => {},
    'pipeline-set-topology': (pipe, topology) => {},
    'pipeline-set-cull': (pipe, cullMode) => {},
    'pipeline-build': (pipe, dev) => 0, // TODO
    'pipeline-destroy': (pipe) => freeHandle(pipe),

    'cmd-create': () => allocHandle({ type: 'cmdEncoder', passes: [] }),
    'cmd-begin-pass': () => {},
    'cmd-end-pass': () => {},
    'cmd-submit': () => {},
    'cmd-set-pipeline': () => {},
    'cmd-draw': () => {},
    'cmd-draw-indexed': () => {},
    'cmd-set-vertex-buffer': () => {},
    'cmd-set-index-buffer': () => {},
    'cmd-set-bind-group': () => {},
    'cmd-set-viewport': () => {},
    'cmd-set-scissor': () => {},

    // Initialization — must be called before WASM start
    async init() {
      return ensureDevice();
    },
  };
})();

export default GPU_HOST;
