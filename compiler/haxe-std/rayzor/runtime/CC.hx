package rayzor.runtime;

/**
 * TinyCC runtime compiler — compile and execute C code at runtime.
 *
 * Gives Haxe programs direct access to platform APIs (Cocoa, X11, Win32),
 * system libraries, and C code without building an rpkg. Ideal for light
 * programs that need quick native access.
 *
 * ## Quick Start
 *
 * ```haxe
 * var cc = CC.create();
 * cc.compile('long add(long a, long b) { return a + b; }');
 * cc.relocate();
 * var fn = cc.getSymbol("add");
 * trace(CC.call2(fn, 3, 4)); // 7
 * ```
 *
 * ## Shared Environment (Heap Memory Bridge)
 *
 * On ARM64 macOS, JIT code cannot use C `static` variables (W^X
 * restriction — code and data share the same page with mutually
 * exclusive write/execute permissions).
 *
 * Instead, allocate state on the **heap** via `calloc` and pass the
 * pointer between Haxe and C. Heap memory is always writable.
 *
 * ```haxe
 * var cc = CC.create();
 * cc.compile('
 *     #include <stdlib.h>
 *     // Allocate shared env on heap (outside JIT W^X region)
 *     long alloc_env(void) { return (long)calloc(8, sizeof(long)); }
 *
 *     // Store a value — C writes to heap, Haxe can read later
 *     void set_handle(long env, long value) { ((long*)env)[0] = value; }
 *
 *     // Read a value — C reads what Haxe (or another C fn) stored
 *     long get_handle(long env) { return ((long*)env)[0]; }
 * ');
 * cc.relocate();
 *
 * var env = CC.call0(cc.getSymbol("alloc_env"));
 * CC.call2(cc.getSymbol("set_handle"), env, somePtr);
 * var handle = CC.call1(cc.getSymbol("get_handle"), env);
 * ```
 *
 * For platform APIs that need persistent state (e.g., a window handle
 * shared between creation and event polling), use multiple CC contexts
 * with the env pointer passed as a function argument:
 *
 * ```haxe
 * // CC 1: create window, store handle in env
 * var cc1 = CC.create();
 * cc1.addFramework("Cocoa");
 * cc1.compile('... long create_window(long env) { *((long*)env) = window; ... }');
 * cc1.relocate();
 * var env = CC.call0(cc1.getSymbol("alloc_env"));
 * var view = CC.call1(cc1.getSymbol("create_window"), env);
 *
 * // CC 2: poll events, read window from env
 * var cc2 = CC.create();
 * cc2.addFramework("Cocoa");
 * cc2.compile('... long poll(long env) { id win = (id)(*((long*)env)); ... }');
 * cc2.relocate();
 * var pollFn = cc2.getSymbol("poll");
 *
 * while (CC.call1(pollFn, env) != null) { /* render */ }
 * ```
 *
 * ## Platform Frameworks
 *
 * ```haxe
 * cc.addFramework("Cocoa");      // macOS — NSWindow, NSApplication
 * cc.addFramework("Accelerate"); // macOS — vDSP, BLAS, LAPACK
 * cc.addFramework("Metal");      // macOS — GPU compute
 * ```
 *
 * ## System Requirements
 *
 * System headers require platform SDK:
 *   - **macOS**: `xcode-select --install` (CommandLineTools)
 *   - **Linux**: `apt install build-essential`
 *
 * Pure C code (no `#include`) works without any SDK.
 *
 * ## ARM64 macOS Notes
 *
 * - JIT memory uses MAP_JIT with W^X protection
 * - C `static` variables in JIT code cause SIGBUS (write to execute-only page)
 * - Use heap-allocated env (calloc) instead of C statics
 * - Split large C into multiple CC contexts if TCC ARM64 codegen fails
 */
@:native("rayzor::runtime::CC")
extern class CC {
    /**
     * Create a new TCC compilation context.
     * Sets output type to memory (JIT-style).
     */
    @:native("create")
    public static function create():CC;

    /**
     * Compile a string of C source code.
     * Can be called multiple times before relocate().
     *
     * @param code C source code string
     * @return true on success, false on compilation error
     */
    @:native("compile")
    public function compile(code:String):Bool;

    /**
     * Register a symbol (value or pointer) so C code can reference it.
     * C code accesses it via `extern`: `extern long my_sym;`
     *
     * All Haxe reference types (Arc, Vec, Box, class instances) are
     * pointer-sized integers and can be passed directly.
     *
     * @param name Symbol name visible to C code
     * @param value Raw value or pointer address (i64)
     */
    @:native("addSymbol")
    public function addSymbol(name:String, value:Int):Void;

    /**
     * Relocate all compiled code into executable memory.
     * Must be called after all compile() and addSymbol() calls,
     * and before getSymbol().
     *
     * @return true on success, false on relocation error
     */
    @:native("relocate")
    public function relocate():Bool;

    /**
     * Get a function pointer or symbol address by name.
     * Must be called after relocate().
     *
     * @param name Symbol name to look up
     * @return Opaque function pointer (pass to call0/call1/call2/call3)
     */
    @:native("getSymbol")
    public function getSymbol(name:String):rayzor.Ptr<Void>;

    /**
     * Load a macOS framework or shared library so its symbols and headers
     * are available to compiled C code.
     *
     * Frameworks (macOS): `cc.addFramework("Accelerate")` enables
     * `#include <Accelerate/Accelerate.h>` and links framework symbols.
     *
     * Shared libraries: `cc.addFramework("z")` loads libz.dylib/libz.so.
     *
     * @param name Framework or library name (without lib prefix or extension)
     * @return true if loaded successfully
     */
    @:native("addFramework")
    public function addFramework(name:String):Bool;

    /**
     * Add a directory to the include search path.
     *
     * @param path Absolute or relative directory path
     * @return true if path was added successfully
     */
    @:native("addIncludePath")
    public function addIncludePath(path:String):Bool;

    /**
     * Add a C source file, object file, archive, or shared library.
     * Supports `.c`, `.o`, `.a`, `.dylib`/`.so`/`.dll`.
     * Must be called before relocate().
     *
     * @param path Path to the file
     * @return true if file was added successfully
     */
    @:native("addFile")
    public function addFile(path:String):Bool;

    /**
     * Free the TCC compilation context.
     * Relocated code memory remains valid (intentional leak for JIT use).
     */
    @:native("delete")
    public function delete():Void;

    /** Call a JIT function (0 args). All values are pointer-sized (i64). */
    @:native("call0")
    public static function call0(fnAddr:rayzor.Ptr<Void>):rayzor.Ptr<Void>;

    /** Call a JIT function (1 arg). */
    @:native("call1")
    public static function call1(fnAddr:rayzor.Ptr<Void>, arg0:rayzor.Ptr<Void>):rayzor.Ptr<Void>;

    /** Call a JIT function (2 args). */
    @:native("call2")
    public static function call2(fnAddr:rayzor.Ptr<Void>, arg0:rayzor.Ptr<Void>, arg1:rayzor.Ptr<Void>):rayzor.Ptr<Void>;

    /** Call a JIT function (3 args). */
    @:native("call3")
    public static function call3(fnAddr:rayzor.Ptr<Void>, arg0:rayzor.Ptr<Void>, arg1:rayzor.Ptr<Void>, arg2:rayzor.Ptr<Void>):rayzor.Ptr<Void>;
}
