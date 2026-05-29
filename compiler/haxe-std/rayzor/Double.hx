package rayzor;

/**
 * 64-bit IEEE-754 floating-point value — a type-system alias for `Float`
 * that documents author intent to round-trip exactly as f64.
 *
 * # Current implementation
 *
 * Today this is a `typedef` aliasing `Float`. The aliasing is opaque to
 * the user (you can pass `Float` where `Double` is expected and vice
 * versa), so it inherits all of `Float`'s behaviour — including the
 * f64-precision guarantees for `Array<Float>` / `Map<K,Float>` that
 * landed in the audit captured by `bugs_array_map_float_audit`.
 *
 * # Why this exists
 *
 * Haxe's `Float` is *nominally* f64 but historically slipped to f32
 * (or to `Any`-shaped 8-byte slots in some generic containers) on
 * certain targets. `Double` is the type-system signal that "this is
 * f64 and I am relying on it" — useful as:
 *
 * - Documentation in API surfaces that care about precision
 *   (ML model weights, audio buffers, scientific data).
 * - A future forcing function: when Rayzor's abstract-over-primitive
 *   handling stops truncating (see `bugs_abstract_over_float_truncates`),
 *   `Double` will be promoted to `@:notNull abstract Double(Float) from
 *   Float to Float` — nominally distinct, dispatchable separately for
 *   any future `Array<Double>` specialisation (packed `[f64]` storage,
 *   etc.). User code that adopts `Double` today will pick that up
 *   automatically.
 *
 * For ordinary arithmetic, prefer `Float` and only annotate `Double`
 * at storage boundaries (collection writes, tensor parameters, etc.).
 *
 * # Example
 *
 * ```haxe
 * import rayzor.Double;
 *
 * function storeWeights(buf:Array<Double>):Void { ... }
 *
 * var w:Double = 3.14;
 * storeWeights([w, 2.718]);
 * ```
 */
typedef Double = Float;
