package rayzor.gpu;

/** Texture pixel format. Indices match Rust `texture_format_from_int`. */
enum TextureFormat {
    BGRA8Unorm;        // 0
    RGBA8Unorm;        // 1
    Depth24PlusStencil8; // 2
    Depth32Float;      // 3
    RGBA16Float;       // 4
    RGBA32Float;       // 5
    BGRA8UnormSrgb;    // 6
    RGBA8UnormSrgb;    // 7
}

/** Primitive topology for rasterization. */
enum PrimitiveTopology {
    TriangleList;      // 0
    TriangleStrip;     // 1
    LineList;          // 2
    LineStrip;         // 3
    PointList;         // 4
}

/** Face culling mode. */
enum CullMode {
    None;   // 0
    Front;  // 1
    Back;   // 2
}

/** Winding order for front-facing triangles. */
enum FrontFace {
    CCW;    // 0
    CW;     // 1
}

/** Depth/stencil comparison function. */
enum CompareFunction {
    Never;          // 0
    Less;           // 1
    Equal;          // 2
    LessEqual;      // 3
    Greater;        // 4
    NotEqual;       // 5
    GreaterEqual;   // 6
    Always;         // 7
}

/** Render pass load operation. */
enum LoadOp {
    Clear;  // 0 — clear to a color
    Load;   // 1 — preserve existing content
}

/** Render pass store operation. */
enum StoreOp {
    Store;   // 0 — write results to texture
    Discard; // 1 — discard results
}

/** Index buffer element format. */
enum IndexFormat {
    Uint16;  // 0
    Uint32;  // 1
}

/** Texture filter mode. */
enum FilterMode {
    Nearest; // 0
    Linear;  // 1
}

/** Texture address (wrap) mode. */
enum AddressMode {
    ClampToEdge;   // 0
    Repeat;        // 1
    MirrorRepeat;  // 2
}

/** Vertex attribute data format. */
enum VertexFormat {
    Float32;       // 0
    Float32x2;     // 1
    Float32x3;     // 2
    Float32x4;     // 3
    Sint32;        // 4
    Uint32;        // 5
}

/** RGBA color with floating-point components [0.0 – 1.0]. */
typedef Color = {
    r:Float,
    g:Float,
    b:Float,
    a:Float,
};

/** A single vertex attribute in a vertex buffer layout. */
typedef VertexAttribute = {
    format:VertexFormat,
    offset:Int,
    shaderLocation:Int,
};

/** Describes the layout of a vertex buffer (stride + attributes). */
typedef VertexBufferLayout = {
    arrayStride:Int,
    attributes:Array<VertexAttribute>,
};

/** Color target state for a render pipeline. */
typedef ColorTargetState = {
    format:TextureFormat,
};

/** Describes a color attachment for a render pass. */
typedef RenderPassColorAttachment = {
    view:Dynamic,
    loadOp:LoadOp,
    storeOp:StoreOp,
    clearColor:Color,
};

/** Buffer usage flags (can be combined with |). */
class BufferUsage {
    public static inline var VERTEX:Int   = 1;
    public static inline var INDEX:Int    = 2;
    public static inline var UNIFORM:Int  = 4;
    public static inline var STORAGE:Int  = 8;
    public static inline var COPY_SRC:Int = 16;
    public static inline var COPY_DST:Int = 32;
}

/** Texture usage flags (can be combined with |). */
class TextureUsage {
    public static inline var COPY_SRC:Int         = 1;
    public static inline var COPY_DST:Int         = 2;
    public static inline var TEXTURE_BINDING:Int  = 4;
    public static inline var STORAGE_BINDING:Int  = 8;
    public static inline var RENDER_ATTACHMENT:Int = 16;
}
