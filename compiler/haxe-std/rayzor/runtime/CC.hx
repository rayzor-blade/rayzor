package rayzor.runtime;

/**
 * TinyCC runtime compiler — compile and execute C code at runtime.
 *
 * Standard C library functions (strlen, printf, malloc, sin, etc.) are
 * available automatically. System headers like `<string.h>`, `<stdio.h>`,
 * and `<math.h>` can be used via `#include`.
 *
 * ## System Requirements
 *
 * System headers require platform SDK headers to be installed:
 *   - **macOS**: Install CommandLineTools via `xcode-select --install` (~1GB).
 *     Full Xcode.app also works but is not required.
 *   - **Linux**: Install build-essential (or equivalent) for `/usr/include`.
 *
 * Pure C code (no `#include`) and `@:cstruct` interop work without any SDK.
 *
 * ## Usage
 *
 * Explicit CC API:
 * ```haxe
 * var cc = CC.create();
 * cc.compile('
 *     #include <string.h>
 *     long my_strlen(long addr) {
 *         return (long)strlen((const char*)addr);
 *     }
 * ');
 * cc.relocate();
 * var fn = cc.getSymbol("my_strlen");
 * trace(CC.call1(fn, cs.raw()));
 * cc.delete();
 * ```
 *
 * Inline `__c__()` syntax (auto-manages TCC lifecycle):
 * ```haxe
 * var len = untyped __c__('
 *     #include <string.h>
 *     long __entry__() {
 *         return (long)strlen((const char*){0});
 *     }
 * ', cs.raw());
 * ```
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
     * @return Function pointer (pass to call0/call1/call2/call3)
     */
    @:native("getSymbol")
    public function getSymbol(name:String):Dynamic;

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
     * Allows `#include "header.h"` and `#include <header.h>` to find
     * headers in the specified directory.
     *
     * @param path Absolute or relative directory path
     * @return true if path was added successfully
     */
    @:native("addIncludePath")
    public function addIncludePath(path:String):Bool;

    /**
     * Add a C source file, object file, archive, or shared library.
     * Supports `.c` (compiled into context), `.o`, `.a` (linked),
     * and `.dylib`/`.so`/`.dll` (dynamically loaded).
     *
     * Must be called before relocate().
     *
     * @param path Path to the file
     * @return true if file was added successfully
     */
    @:native("addFile")
    public function addFile(path:String):Bool;

    /**
     * Free the TCC compilation context and all associated resources.
     * Note: relocated code memory remains valid (intentional leak for JIT use).
     */
    @:native("delete")
    public function delete():Void;

    /**
     * Call a JIT-compiled function (0 args) by its address.
     * @param fnAddr Function pointer from getSymbol()
     * @return Function return value as Int
     */
    @:native("call0")
    public static function call0(fnAddr:Int):Int;

    /**
     * Call a JIT-compiled function (1 arg) by its address.
     */
    @:native("call1")
    public static function call1(fnAddr:Int, arg0:Int):Int;

    /**
     * Call a JIT-compiled function (2 args) by its address.
     */
    @:native("call2")
    public static function call2(fnAddr:Int, arg0:Int, arg1:Int):Int;

    /**
     * Call a JIT-compiled function (3 args) by its address.
     */
    @:native("call3")
    public static function call3(fnAddr:Int, arg0:Int, arg1:Int, arg2:Int):Int;
}
