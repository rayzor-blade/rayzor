package nue.loader;

import haxe.io.Bytes;
import haxe.ds.StringMap;

/**
 * Low-level GGUF binary reader. Parses the header, KV metadata, and
 * tensor info table — does NOT decode tensor data into Tensors (that
 * is the loader's job, and the dtype determines whether the bytes
 * become a Tensor or a QTensor).
 *
 * GGUF v3 layout:
 * ```
 *   magic        : 4 bytes "GGUF"
 *   version      : u32
 *   n_tensors    : u64
 *   n_meta_kv    : u64
 *   meta_kv      : [(key:gstring, type:u32, value:dep)] * n_meta_kv
 *   tensor_infos : [(name:gstring, n_dims:u32, dims:[u64]*n,
 *                    dtype:u32, offset:u64)] * n_tensors
 *   <pad to alignment (default 32 bytes)>
 *   tensor_data  : raw bytes
 * ```
 * All integers are little-endian. Strings are length-prefixed u64
 * followed by UTF-8 bytes (no terminator). KV value layout depends
 * on the type tag — see `MetaValue` below.
 *
 * **Storage choice.** Metadata is held in a `StringMap<MetaValue>`
 * keyed by the GGUF KV key. O(1) average lookup; insertion order is
 * not preserved (and not needed — GGUF consumers always know the key).
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */
class GGUFReader {
    public static inline var MAGIC:Int = 0x46554747; // "GGUF" little-endian
    public static inline var ALIGNMENT_DEFAULT:Int = 32;

    public var bytes:Bytes;
    public var version:Int;
    public var nTensors:Int;
    public var nMetaKv:Int;
    public var meta:StringMap<MetaValue>;
    public var tensorInfos:Array<TensorInfo>;
    /** Byte offset where the tensor data section begins. */
    public var dataStart:Int;
    /** Alignment from `general.alignment` meta (defaults to 32). */
    public var alignment:Int;

    private var pos:Int;

    public function new(bytes:Bytes) {
        this.bytes = bytes;
        this.pos = 0;
        this.meta = new StringMap<MetaValue>();
        this.tensorInfos = [];
        parse();
    }

    /** O(1) lookup of a metadata value. Returns `null` if absent. */
    public function findMeta(key:String):MetaValue {
        return meta.get(key);
    }

    private function parse():Void {
        var magic = bytes.getInt32(0);
        if (magic != MAGIC) {
            throw "GGUFReader: bad magic 0x" + StringTools.hex(magic) + " (expected 'GGUF')";
        }
        version = bytes.getInt32(4);
        if (version < 2 || version > 3) {
            throw "GGUFReader: unsupported GGUF version " + version + " (need 2 or 3)";
        }
        // GGUF v3 header layout:
        //   offset  0..3   magic ("GGUF")
        //   offset  4..7   version (u32)
        //   offset  8..15  n_tensors (u64 — low 32 bits suffice)
        //   offset 16..23  n_metadata_kv (u64 — low 32 bits suffice)
        //   offset 24..    metadata KV pairs
        nTensors = bytes.getInt32(8);
        nMetaKv = bytes.getInt32(16);
        pos = 24;

        for (_ in 0...nMetaKv) {
            var key = readString();
            var typeTag = readU32();
            var value = readValue(typeTag);
            meta.set(key, value);
        }

        for (_ in 0...nTensors) {
            var name = readString();
            var nDims = readU32();
            var dims:Array<Int> = [];
            for (_ in 0...nDims) {
                dims.push(readU32());
                pos += 4;
            }
            var dtype = readU32();
            var offsetLo = bytes.getInt32(pos);
            var offsetHi = bytes.getInt32(pos + 4);
            pos += 8;
            tensorInfos.push(new TensorInfo(name, dims, dtype, offsetLo, offsetHi));
        }

        alignment = ALIGNMENT_DEFAULT;
        var alignMeta = findMeta("general.alignment");
        if (alignMeta != null) {
            switch (alignMeta) {
                case U32(v): alignment = v;
                case I32(v): alignment = v;
                case _:
            }
        }

        var rem = pos % alignment;
        if (rem != 0) pos += alignment - rem;
        dataStart = pos;
    }

    private function readString():String {
        var len = readU64Truncated();
        // `Bytes.getString` isn't exposed by the rayzor stdlib (it lives
        // in `haxe.io.Bytes`'s pure-Haxe body which the rayzor typedef
        // bypasses). `sub(pos, len).toString()` is the same operation and
        // dispatches through real runtime entries.
        var s = bytes.sub(pos, len).toString();
        pos += len;
        return s;
    }

    private inline function readU32():Int {
        var v = bytes.getInt32(pos);
        pos += 4;
        return v;
    }

    /** Read a u64 and truncate to i32 (sufficient for shapes/counts). */
    private inline function readU64Truncated():Int {
        var lo = bytes.getInt32(pos);
        pos += 8;
        return lo;
    }

    private function readValue(typeTag:Int):MetaValue {
        switch (typeTag) {
            case 0: var v = bytes.get(pos); pos += 1; return U8(v);
            case 1: var v = bytes.get(pos); pos += 1; return I8(v);
            case 2: var v = bytes.getUInt16(pos); pos += 2; return U16(v);
            case 3: var v = bytes.getUInt16(pos); pos += 2; return I16(v);
            case 4: return U32(readU32());
            case 5: return I32(readU32());
            case 6: var v = bytes.getFloat(pos); pos += 4; return F32(v);
            case 7: var v = bytes.get(pos); pos += 1; return Bool(v != 0);
            case 8: return Str(readString());
            case 9: return readArray();
            case 10: return U64(readU64Truncated());
            case 11: return I64(readU64Truncated());
            case 12: var v = bytes.getDouble(pos); pos += 8; return F64(v);
            case _: throw "GGUFReader: unknown meta value type " + typeTag;
        }
    }

    private function readArray():MetaValue {
        var elemType = readU32();
        var count = readU64Truncated();
        var items:Array<MetaValue> = [];
        for (_ in 0...count) {
            items.push(readValue(elemType));
        }
        return Arr(elemType, items);
    }

    /** Return the raw bytes of the tensor at `info`. The slice is
        a view into the underlying file Bytes — do not free.

        Uses `bytes.subWithBase` so the combined `dataStart + offset` can
        exceed 2 GiB without sign-extending into negative i32 territory.
        The runtime handles the u64 add-with-carry; we just pass the three
        components through unchanged. **/
    public function tensorBytes(info:TensorInfo):Bytes {
        var size = tensorByteSize(info);
        return bytes.subWithBase(dataStart, info.offsetLo, info.offsetHi, size);
    }

    /** Compute the on-disk byte size of a tensor (dtype-dependent). */
    public static function tensorByteSize(info:TensorInfo):Int {
        var nElem = 1;
        for (d in info.dims) nElem *= d;
        return switch (info.dtype) {
            case 0: nElem * 4;       // F32
            case 1: nElem * 2;       // F16
            case 8: nElem + Std.int(nElem / 32) * 2;   // Q8_0
            case 12: blockedSize(nElem, 256, 144);     // Q4_K — 144-byte super-blocks
            case 14: blockedSize(nElem, 256, 210);     // Q6_K — 210-byte super-blocks
            case _: throw "GGUFReader: byte-size for dtype " + info.dtype + " not implemented";
        };
    }

    private static inline function blockedSize(nElem:Int, blockElems:Int, blockBytes:Int):Int {
        var nBlocks = Std.int(nElem / blockElems);
        if (nElem % blockElems != 0) nBlocks += 1;
        return nBlocks * blockBytes;
    }

    /** Convenience: read an Int meta value. Throws if missing or wrong type. */
    public function metaInt(key:String):Int {
        var v = findMeta(key);
        if (v == null) throw "GGUFReader: missing meta key '" + key + "'";
        return switch (v) {
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: throw "GGUFReader: meta key '" + key + "' is not an integer";
        };
    }

    /** Convenience: read a Float meta value. Throws if missing or wrong type. */
    public function metaFloat(key:String):Float {
        var v = findMeta(key);
        if (v == null) throw "GGUFReader: missing meta key '" + key + "'";
        return switch (v) {
            case F32(x) | F64(x): x;
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: throw "GGUFReader: meta key '" + key + "' is not a float";
        };
    }

    public function metaString(key:String):String {
        var v = findMeta(key);
        if (v == null) throw "GGUFReader: missing meta key '" + key + "'";
        return switch (v) {
            case Str(s): s;
            case _: throw "GGUFReader: meta key '" + key + "' is not a string";
        };
    }

    /**
     * Return the items of an `Arr` metadata value. Throws if the key
     * is missing or the value isn't an array. Each element is still a
     * `MetaValue` — the caller pattern-matches on the expected element
     * type (e.g., `Str` for token strings).
     */
    public function metaArray(key:String):Array<MetaValue> {
        var v = findMeta(key);
        if (v == null) throw "GGUFReader: missing meta key '" + key + "'";
        return switch (v) {
            case Arr(_, items): items;
            case _: throw "GGUFReader: meta key '" + key + "' is not an array";
        };
    }

    /** Convenience: same as `metaArray` but returns `null` instead of throwing. */
    public function metaArrayOr(key:String):Array<MetaValue> {
        var v = findMeta(key);
        if (v == null) return null;
        return switch (v) {
            case Arr(_, items): items;
            case _: null;
        };
    }
}

/**
 * Per-tensor index entry: name, shape (dims), GGML dtype code, and
 * byte offset within the file's data region.
 *
 * Modelled as a class (not a `typedef = {...}` anon struct) so the
 * per-field offsets stay stable when an `Array<TensorInfo>` crosses
 * compilation contexts. The anon-struct read path scrambles field
 * indices cross-file (see `bugs_anon_field_scramble`); class field
 * accesses go through a fixed GEP layout that survives BLADE
 * save/load.
 */
class TensorInfo {
    public var name:String;
    public var dims:Array<Int>;
    public var dtype:Int;
    /** Low 32 bits of the u64 byte offset within the file's data region. */
    public var offsetLo:Int;
    /** High 32 bits of the u64 byte offset. Non-zero for files >4 GiB or
        for tensors past the 2 GiB mark when the low 32 bits sign-extend. */
    public var offsetHi:Int;
    public function new(name:String, dims:Array<Int>, dtype:Int, offsetLo:Int, offsetHi:Int) {
        this.name = name;
        this.dims = dims;
        this.dtype = dtype;
        this.offsetLo = offsetLo;
        this.offsetHi = offsetHi;
    }
}

/**
 * Tagged-union encoding of every GGUF metadata value. Pattern-match
 * at the call site rather than try to coerce — model files use
 * mixed-width integers liberally and the consumer always knows what
 * type a given key should be.
 */
enum MetaValue {
    U8(v:Int);
    I8(v:Int);
    U16(v:Int);
    I16(v:Int);
    U32(v:Int);
    I32(v:Int);
    U64(v:Int);
    I64(v:Int);
    F32(v:Float);
    F64(v:Float);
    Bool(v:Bool);
    Str(v:String);
    Arr(elemType:Int, items:Array<MetaValue>);
}
