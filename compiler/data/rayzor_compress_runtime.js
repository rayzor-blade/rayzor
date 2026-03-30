// Rayzor Compress Runtime — synchronous deflate/inflate
// MVP: copy-through for streaming, CompressionStream for one-shot
// TODO: embed pako-lite for real DEFLATE

const _cmp = new Map();
let _cmpN = 1;

const compressRuntime = {
  rayzor_compress_new(level) {
    const h = _cmpN++;
    _cmp.set(h, { level: Math.max(0, Math.min(9, level)), flush: 0, done: false });
    return h;
  },
  rayzor_compress_execute(h, srcBytesPtr, srcPos, dstBytesPtr, dstPos) {
    // MVP: copy src→dst (no actual deflate). Returns {done, read, write} struct.
    if (!memory) return 0;
    const c = _cmp.get(h); if (!c) return 0;
    const v = new DataView(memory.buffer);
    // HaxeBytes: {ptr: u32, len: u32}
    const srcDataPtr = v.getUint32(srcBytesPtr, true);
    const srcLen = v.getUint32(srcBytesPtr + 4, true);
    const dstDataPtr = v.getUint32(dstBytesPtr, true);
    const dstLen = v.getUint32(dstBytesPtr + 4, true);
    const avail = Math.min(srcLen - srcPos, dstLen - dstPos);
    new Uint8Array(memory.buffer).copyWithin(dstDataPtr + dstPos, srcDataPtr + srcPos, srcDataPtr + srcPos + avail);
    const done = c.flush === 3 ? 1 : 0;
    // Return anonymous {done:Bool, read:Int, write:Int} as flat struct
    const rPtr = malloc(24);
    v.setInt32(rPtr, done, true);     // done
    v.setInt32(rPtr + 8, avail, true);  // read
    v.setInt32(rPtr + 16, avail, true); // write
    return rPtr;
  },
  rayzor_compress_set_flush(h, mode) { const c = _cmp.get(h); if (c) c.flush = mode; },
  rayzor_compress_close(h) { _cmp.delete(h); },
  rayzor_compress_run(srcBytesPtr, level) {
    // One-shot compress: return copy (MVP — no actual deflate)
    if (!memory) return 0;
    const v = new DataView(memory.buffer);
    const dataPtr = v.getUint32(srcBytesPtr, true);
    const len = v.getUint32(srcBytesPtr + 4, true);
    const dstPtr = malloc(len);
    new Uint8Array(memory.buffer).copyWithin(dstPtr, dataPtr, dataPtr + len);
    // Return HaxeBytes {ptr, len}
    const bytesPtr = malloc(8);
    v.setUint32(bytesPtr, dstPtr, true);
    v.setUint32(bytesPtr + 4, len, true);
    return bytesPtr;
  },
  rayzor_uncompress_new(windowBits) {
    const h = _cmpN++;
    _cmp.set(h, { flush: 0, done: false });
    return h;
  },
  rayzor_uncompress_execute(h, src, srcPos, dst, dstPos) {
    return compressRuntime.rayzor_compress_execute(h, src, srcPos, dst, dstPos);
  },
  rayzor_uncompress_set_flush(h, mode) { compressRuntime.rayzor_compress_set_flush(h, mode); },
  rayzor_uncompress_close(h) { compressRuntime.rayzor_compress_close(h); },
};
