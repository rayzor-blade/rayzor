package rayzor.ds;

/**
 * N-dimensional tensor with shape, strides, and dtype.
 *
 * Tensor is the fundamental data structure for numerical computing in Rayzor.
 * It supports arbitrary shapes, strided views (reshape/transpose without copy),
 * and element-wise + linear algebra operations.
 *
 * The runtime representation is a heap-allocated struct with reference counting
 * for shared views. All operations return new tensors (immutable value semantics).
 *
 * Example:
 * ```haxe
 * var a = Tensor.zeros([2, 3], F32);
 * var b = Tensor.ones([2, 3], F32);
 * var c = a + b;
 * trace(c.sum());
 * ```
 */
@:native("rayzor::ds::Tensor")
extern class Tensor {
    // --- Construction ---

    /** Create a tensor filled with zeros */
    @:native("tensor_zeros")
    public static function zeros(shape:Array<Int>, dtype:DType):Tensor;

    /** Create a tensor filled with ones */
    @:native("tensor_ones")
    public static function ones(shape:Array<Int>, dtype:DType):Tensor;

    /** Create a tensor filled with a constant value */
    @:native("tensor_full")
    public static function full(shape:Array<Int>, value:Float, dtype:DType):Tensor;

    /** Create a 1-D tensor from a flat array of floats */
    @:native("tensor_fromArray")
    public static function fromArray(data:Array<Float>, dtype:DType):Tensor;

    /** Create a tensor with random values in [0, 1) */
    @:native("tensor_rand")
    public static function rand(shape:Array<Int>, dtype:DType):Tensor;

    // --- Properties ---

    /** Get the shape as an array of dimension sizes */
    @:native("tensor_shape")
    public function shape():Array<Int>;

    /** Number of dimensions */
    @:native("tensor_ndim")
    public function ndim():Int;

    /** Total number of elements */
    @:native("tensor_numel")
    public function numel():Int;

    /** Element data type */
    @:native("tensor_dtype")
    public function dtype():DType;

    // --- Element access ---

    /** Get element at indices */
    @:native("tensor_get")
    public function get(indices:Array<Int>):Float;

    /** Set element at indices */
    @:native("tensor_set")
    public function set(indices:Array<Int>, value:Float):Void;

    // --- Reshape / view (no copy) ---

    /** Reshape to a new shape (same numel) */
    @:native("tensor_reshape")
    public function reshape(shape:Array<Int>):Tensor;

    /** 2D matrix transpose */
    @:native("tensor_transpose")
    public function transpose():Tensor;

    /** N-D permutation (no copy, view) */
    @:native("tensor_permute")
    public function permute(axes:Array<Int>):Tensor;

    /** Slice along a single dim, [start, end) (no copy, view) */
    @:native("tensor_slice")
    public function slice(dim:Int, start:Int, end:Int):Tensor;

    // --- Arithmetic (elementwise, return new tensor) ---

    /** Element-wise addition */
    @:native("tensor_add")
    @:op(A + B)
    public function add(other:Tensor):Tensor;

    /** Element-wise subtraction */
    @:native("tensor_sub")
    @:op(A - B)
    public function sub(other:Tensor):Tensor;

    /** Element-wise multiplication */
    @:native("tensor_mul")
    @:op(A * B)
    public function mul(other:Tensor):Tensor;

    /** Element-wise division */
    @:native("tensor_div")
    @:op(A / B)
    public function div(other:Tensor):Tensor;

    // --- Linear algebra ---

    /** Matrix multiplication */
    @:native("tensor_matmul")
    public function matmul(other:Tensor):Tensor;

    /** Dot product (flattened) */
    @:native("tensor_dot")
    public function dot(other:Tensor):Float;

    // --- Reductions ---

    /** Sum all elements (returns scalar tensor) */
    @:native("tensor_sum")
    public function sum():Float;

    /** Mean of all elements */
    @:native("tensor_mean")
    public function mean():Float;

    /** Maximum element */
    @:native("tensor_max")
    public function max():Float;

    /** Minimum element */
    @:native("tensor_min")
    public function min():Float;

    // --- Math ---

    /** Element-wise square root */
    @:native("tensor_sqrt")
    public function sqrt():Tensor;

    /** Element-wise exponential */
    @:native("tensor_exp")
    public function exp():Tensor;

    /** Element-wise natural logarithm */
    @:native("tensor_log")
    public function log():Tensor;

    /** Element-wise ReLU activation */
    @:native("tensor_relu")
    public function relu():Tensor;

    /** Element-wise GELU activation (tanh approximation) */
    @:native("tensor_gelu")
    public function gelu():Tensor;

    /** Element-wise SiLU / swish activation */
    @:native("tensor_silu")
    public function silu():Tensor;

    /** Softmax over the last dimension */
    @:native("tensor_softmax")
    public function softmax():Tensor;

    /** Layer normalization over the last dimension */
    @:native("tensor_layer_norm")
    public function layerNorm(eps:Float):Tensor;

    /** RMS normalization over the last dimension */
    @:native("tensor_rms_norm")
    public function rmsNorm(eps:Float):Tensor;

    // --- Interop ---

    /** Get raw data pointer for FFI */
    @:native("tensor_data")
    public function data():Ptr<Float>;

    /** Free tensor and its data */
    @:native("tensor_free")
    public function free():Void;
}
