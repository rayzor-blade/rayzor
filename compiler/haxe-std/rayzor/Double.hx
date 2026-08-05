package rayzor;

/**
 * 64-bit IEEE-754 floating-point value — a type-system alias for `Float`
 * that documents author intent to round-trip exactly as f64.
 *
 * # Current implementation
 *
 * Today this is a `typedef` aliasing `Float`. The aliasing is opaque to
 * the user (you can pass `Float` where `Double` is expected and vice
 * versa), so it inherits all of `Float`'s behaviour, including full
 * f64 precision in `Array<Float>` and `Map<K,Float>`.
 *
 * # Why this exists
 *
 * `Float` is f64, but nothing in the type carries the author's *intent*
 * to depend on that. `Double` is the type-system signal that "this is
 * f64 and I am relying on it" — useful as:
 *
 * - Documentation in API surfaces that care about precision
 *   (ML model weights, audio buffers, scientific data).
 * - A forcing function: `Double` is reserved to become
 *   `@:notNull abstract Double(Float) from Float to Float` — nominally
 *   distinct, and dispatchable separately for any future `Array<Double>`
 *   specialisation (packed `[f64]` storage, etc.). Code that adopts
 *   `Double` today picks that up automatically.
 *
 * For single precision, use `Single` (f32) instead.
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
