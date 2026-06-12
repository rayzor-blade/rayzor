import sys.io.File;
import nue.loader.GGUFLoader;
import nue.loader.GGUFReader;
import nue.loader.GGUFReader.MetaValue;

class Main {
    static function main() {
        var args = Sys.args();
        if (args.length == 0) {
            trace("missing gguf path");
            return;
        }
        var bytes = File.getBytes(args[0]);
        trace("magic=" + StringTools.hex(bytes.getInt32(0)));
        trace("version=" + bytes.getInt32(4));
        trace("tensors=" + bytes.getInt32(8));
        trace("metaKvs=" + bytes.getInt32(16));
        var reader = new GGUFReader(bytes);
        trace("helper.first=" + archFromReader(reader));
        trace("reader.version=" + reader.version);
        trace("reader.nTensors=" + reader.nTensors);
        trace("reader.nMetaKv=" + reader.nMetaKv);
        trace("arch=" + reader.metaString("general.architecture"));
        trace("embed=" + reader.metaInt("llama.embedding_length"));
        trace("embed.direct.u32=" + directMetaU32(reader.findMeta("llama.embedding_length")));
        trace("embed.direct=" + directMetaInt(reader.findMeta("llama.embedding_length")));
        trace("helper.arch=" + archFromReader(reader));
        var loader = new GGUFLoader();
        var meta = loader.readMetadata(args[0]);
        trace("loader.arch=" + meta.architecture);
        trace("loader.hidden=" + meta.hiddenSize);
        var mini = new MiniLoader();
        trace("mini.arch=" + mini.loadWithTokenizerShape(args[0], 4096));
        if (args.length > 1 && args[1] == "load") {
            trace("real.load.start");
            var loaded = loader.loadWithTokenizer(args[0], 4096);
            trace("real.load.arch=" + loaded.metadata.architecture);
        }
    }

    static function archFromReader(reader:GGUFReader):String {
        return reader.metaString("general.architecture");
    }

    static function directMetaInt(v:MetaValue):Int {
        if (v == null) return -99;
        return switch (v) {
            case U8(x) | I8(x) | U16(x) | I16(x) | U32(x) | I32(x) | U64(x) | I64(x): x;
            case _: -1;
        };
    }

    static function directMetaU32(v:MetaValue):Int {
        if (v == null) return -99;
        return switch (v) {
            case U32(x): x;
            case _: -1;
        };
    }
}

class MiniLoader {
    public function new() {}

    public function loadWithTokenizerShape(path:String, maxCtx:Int = 0):String {
        trace("[mini] 1");
        var bytes = File.getBytes(path);
        trace("[mini] 2");
        var reader = new GGUFReader(bytes);
        trace("[mini] 3");
        var arch = metadataArch(reader);
        if (maxCtx > 0) {
            trace("[mini] maxCtx=" + maxCtx);
        }
        trace("[mini] 4");
        return arch;
    }

    private static function metadataArch(reader:GGUFReader):String {
        return reader.metaString("general.architecture");
    }
}
