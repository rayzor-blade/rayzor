package nue.arch;

/**
 * The structure and policy a model was actually built with, written down.
 *
 * Fusion and dispatch are decided in the kernels today, from environment flags
 * read at their first call site. That works, but it leaves nothing to read: a
 * run cannot say which route it took, and a change cannot be shown to have
 * preserved it. This records the decisions once, where the model is assembled
 * and every input is already in scope, so a run can be compared against another
 * run rather than against a recollection.
 *
 * It observes; nothing consults it. That is deliberate for a first step — the
 * plan has to be shown to describe today's behaviour before anything is allowed
 * to depend on it.
 *
 * Nodes are held as parallel arrays of primitives rather than a node class:
 * this object is built, printed and dropped inside one function, and primitives
 * keep it free of ownership questions.
 */
class NuePlan {
    // Node kinds. Private on purpose: they are compared only inside this file,
    // so nothing outside can come to depend on their values.
    static inline var OP_EMBED = 0;
    static inline var OP_ATTN_NORM = 1;
    static inline var OP_ATTENTION = 2;
    static inline var OP_FFN_NORM = 3;
    static inline var OP_FFN = 4;
    static inline var OP_OUT_NORM = 5;
    static inline var OP_LM_HEAD = 6;

    // Which pass a node belongs to. Prefill and decode take different routes
    // through the same modules, so a node that exists in both is recorded once
    // and marked here.
    static inline var PHASE_BOTH = 0;

    // Weight schemes, as ordinals so this class needs no tensor types.
    public static inline var SCHEME_NONE = -1;
    public static inline var SCHEME_INT8 = 0;
    public static inline var SCHEME_Q4_K_M = 1;
    public static inline var SCHEME_Q6_K = 2;
    public static inline var SCHEME_Q8_0 = 3;

    // Cache kinds, as built rather than as requested — a Q8 cache downgrades
    // when the head dimension is not a multiple of 32, and again if the plugin
    // allocation fails, so what was asked for is not what attention branches on.
    public static inline var CACHE_F32 = 0;
    public static inline var CACHE_Q8 = 1;
    public static inline var CACHE_Q8_HAXE = 2;

    // ---- policy ----------------------------------------------------------
    // Every field is a primitive, and every one is filled from a value the
    // builder already had. Nothing here reads a flag that the build did not
    // read anyway.
    public var haxeMatmul:Bool = false;
    public var kvQ8:Bool = false;
    public var haxeFlash:Bool = false;
    public var requantLmHead:Bool = false;
    public var requantQ6K:Bool = false;
    public var poolWorkers:Int = 0;
    public var poolSpins:Int = 0;
    public var poolRelax:Int = 0;
    public var poolProfiling:Bool = false;
    public var poolAdaptive:Bool = false;

    // ---- shape -----------------------------------------------------------
    public var architecture:String = "";
    public var layers:Int = 0;
    public var heads:Int = 0;
    public var kvHeads:Int = 0;
    public var headDim:Int = 0;
    public var ropeNeox:Bool = false;
    public var lmHeadScheme:Int = SCHEME_NONE;

    // ---- nodes -----------------------------------------------------------
    // Parallel arrays: `op[i]` is the kind, `layer[i]` the decoder layer it
    // belongs to (-1 for the ones outside the stack), and `a`/`b`/`c` carry
    // whatever that kind needs — for attention, the query/key/value schemes.
    public var op:Array<Int> = [];
    public var phase:Array<Int> = [];
    public var layer:Array<Int> = [];
    public var a:Array<Int> = [];
    public var b:Array<Int> = [];
    public var c:Array<Int> = [];
    public var label:Array<String> = [];

    public function new() {}

    public function addNode(op:Int, phase:Int, layer:Int, a:Int, b:Int, c:Int, label:String):Void {
        this.op.push(op);
        this.phase.push(phase);
        this.layer.push(layer);
        this.a.push(a);
        this.b.push(b);
        this.c.push(c);
        this.label.push(label);
    }

    /** One decoder layer: its norms, its attention with the schemes its
        projections were built from and the cache it actually got, and its
        feed-forward. */
    public function addLayer(index:Int, qScheme:Int, kScheme:Int, vScheme:Int, cache:Int):Void {
        addNode(OP_ATTN_NORM, PHASE_BOTH, index, 0, 0, 0, "rmsnorm");
        addNode(OP_ATTENTION, PHASE_BOTH, index, qScheme, kScheme, vScheme, cacheName(cache));
        addNode(OP_FFN_NORM, PHASE_BOTH, index, 0, 0, 0, "rmsnorm");
        addNode(OP_FFN, PHASE_BOTH, index, 0, 0, 0, "swiglu");
    }

    public function addEmbedding(scheme:Int):Void {
        addNode(OP_EMBED, PHASE_BOTH, -1, scheme, 0, 0, "embedding");
    }

    public function addHead(scheme:Int):Void {
        addNode(OP_OUT_NORM, PHASE_BOTH, -1, 0, 0, 0, "rmsnorm");
        addNode(OP_LM_HEAD, PHASE_BOTH, -1, scheme, 0, 0, "linear");
    }

    /** Print the plan. The prefix is `[nue-graph]`; `[nue-plan]` already
        belongs to the kernel census and the two must stay separable. */
    public function dump():Void {
        Sys.println("[nue-graph] model: arch=" + architecture
            + " layers=" + layers
            + " heads=" + heads + "/" + kvHeads
            + " headDim=" + headDim
            + " rope=" + (ropeNeox ? "neox" : "norm")
            + " lm_head=" + schemeName(lmHeadScheme));
        Sys.println("[nue-graph] policy: haxe_matmul=" + onOff(haxeMatmul)
            + " kv_q8=" + onOff(kvQ8)
            + " haxe_flash=" + onOff(haxeFlash)
            + " requant_lm_head=" + onOff(requantLmHead)
            + " requant_q6k=" + onOff(requantQ6K));
        Sys.println("[nue-graph] pool: workers=" + poolWorkers
            + " spins=" + poolSpins
            + " relax=" + poolRelax
            + " profiling=" + onOff(poolProfiling)
            + " adaptive=" + onOff(poolAdaptive));
        // The fused prefill graph is attached by the loader after the model is
        // built, so it is not knowable here.
        Sys.println("[nue-graph] engine: graph_prefill=deferred");

        var nodes = op.length;
        var decoderLayers = 0;
        var i = 0;
        while (i < nodes) {
            if (op[i] == OP_ATTENTION) decoderLayers++;
            i++;
        }
        Sys.println("[nue-graph] nodes: total=" + nodes + " decoder_layers=" + decoderLayers);

        // One line per distinct layer shape rather than per layer: a stack of
        // thirty identical layers says nothing thirty times, and a layer that
        // differs is the thing worth seeing.
        var shapes:Array<String> = [];
        var counts:Array<Int> = [];
        var firsts:Array<Int> = [];
        i = 0;
        while (i < nodes) {
            if (op[i] == OP_ATTENTION) {
                var shape = "q=" + schemeName(a[i]) + " k=" + schemeName(b[i])
                    + " v=" + schemeName(c[i]) + " cache=" + label[i];
                var at = indexOfString(shapes, shape);
                if (at < 0) {
                    shapes.push(shape);
                    counts.push(1);
                    firsts.push(layer[i]);
                } else {
                    counts[at] = counts[at] + 1;
                }
            }
            i++;
        }
        var s = 0;
        while (s < shapes.length) {
            Sys.println("[nue-graph] layer: x" + counts[s] + " from=" + firsts[s] + " " + shapes[s]);
            s++;
        }
    }

    function indexOfString(xs:Array<String>, want:String):Int {
        var i = 0;
        while (i < xs.length) {
            if (xs[i] == want) return i;
            i++;
        }
        return -1;
    }

    function onOff(v:Bool):String {
        return v ? "on" : "off";
    }

    function cacheName(kind:Int):String {
        if (kind == CACHE_Q8_HAXE) return "q8_haxe";
        if (kind == CACHE_Q8) return "q8";
        return "f32";
    }

    function schemeName(ord:Int):String {
        if (ord == SCHEME_INT8) return "int8";
        if (ord == SCHEME_Q4_K_M) return "q4_k_m";
        if (ord == SCHEME_Q6_K) return "q6_k";
        if (ord == SCHEME_Q8_0) return "q8_0";
        return "f32";
    }
}
