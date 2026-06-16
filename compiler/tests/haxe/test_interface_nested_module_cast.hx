interface Module {
    public function forward():String;
}

class Cache {
    public var currentLen:Int;

    public function new(v:Int) {
        currentLen = v;
    }

    public function reset():Void {
        currentLen = 0;
    }
}

class Attention implements Module {
    public var cache:Cache;

    public function new(cache:Cache) {
        this.cache = cache;
    }

    public function forward():String {
        return "attention";
    }
}

class Block implements Module {
    public var attn:Module;

    public function new(attn:Module) {
        this.attn = attn;
    }

    public function forward():String {
        return "block";
    }
}

class Main {
    static function main() {
        var cache = new Cache(7);
        var blocks:Array<Module> = [];
        blocks.push(new Block(new Attention(cache)));

        for (i in 0...blocks.length) {
            var tb = cast(blocks[i], Block);
            var attn = cast(tb.attn, Attention);
            if (attn != null && attn.cache != null) {
                attn.cache.reset();
            }
        }

        trace(cache.currentLen);
    }
}
