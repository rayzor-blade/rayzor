# Rayzor CLI

`rayzor <command> --help` has the full flag list. This page is the short path:
install, then the commands you use daily.

Compiler diagnostics live in [CONTRIBUTING.md](CONTRIBUTING.md): `dump`, `debug`,
stage inspection, and the `RAYZOR_*` variables.

---

## Install

```bash
curl -fsSL https://rayzor.tech/install.sh | sh     # macOS, Linux, WSL
```

```powershell
irm https://rayzor.tech/install.ps1 | iex          # Windows
```

Lands in `~/.rayzor/bin`. Add that to your `PATH` if the installer says so. The
download is self-contained, so there is nothing else to install.

```bash
rayzor info                      # confirm the install
```

## First project

```bash
rayzor init --name my-app        # src/Main.hx + rayzor.toml
cd my-app
rayzor run                       # entry point from the manifest
```

```bash
rayzor init --name lib --template lib          # app | lib | benchmark | empty
rayzor init --name ws --workspace --members a,b
rayzor init --from-hxml build.hxml             # convert an existing Haxe build
```

---

## Daily loop

```bash
rayzor run                       # run the manifest's entry point
rayzor run main.hx               # run one file
rayzor run -- --port 8080        # args after -- go to your program
rayzor check main.hx             # type-check, no codegen
rayzor run --stats               # where compile time went
rayzor run -i                    # TUI afterwards: scroll, search
```

## Shipping

```bash
rayzor aot main.hx -o app        # native binary
rayzor bundle main.hx -o app.rzb # one portable file
rayzor run app.rzb               # ...run it later, no compile step
rayzor build --target wasm       # browser / WASI
rayzor build --target wasm --browser   # + an HTML harness
```

Cross-compiling and stripping:

```bash
rayzor aot main.hx -o app --target aarch64-unknown-linux-gnu
rayzor aot main.hx -o app --strip --strip-symbols
rayzor aot main.hx --emit obj -o app.o        # exe (default) | obj | asm
```

`rayzor build` with no argument resolves an explicit `.hxml`, else `rayzor.toml`,
else the manifest's `hxml = "build.hxml"` delegation. `--dry-run` prints the plan.

## Startup speed

```bash
rayzor run --preset script       # instant start, never promotes
rayzor cache warm                # pre-compile the stdlib once
rayzor run --no-cache            # ignore the cache (it is on by default)
```

| Preset | For |
|---|---|
| `script` | One-shot scripts. Starts instantly, never promotes |
| `application` | Apps and servers. Balanced (**default**) |
| `server` | Long-running services. Optimizes aggressively |
| `benchmark` | Performance testing. Compiles at once, fully optimized within warm-up |
| `development` | Debugging. Verbose logging |
| `embedded` | Constrained targets. Interpreter only |

`run` starts interpreted and promotes a function once it has run enough times; a
preset picks that policy, not a backend.

## Packages

```bash
rayzor rpkg pack                 # build an .rpkg
rayzor rpkg add ./thing.rpkg     # add to this project
rayzor run --rpkg ./thing.rpkg   # ...or load one directly
rayzor run --native-lib ./libplugin.dylib
```

`.rpkg` carries Haxe sources and optionally native dylibs. `rpkg strip` cuts one
down to a single platform; `inspect`, `install`, `remove`, `list` do what they say.

## Editors

```bash
rayzor lsp                       # Language Server
rayzor jit main.hx               # interactive REPL
```

The language server reads class paths, libraries, defines and initialization
macros from HXML. Set `initializationOptions.hxml` to select a build explicitly.
Otherwise it prefers the workspace's `build.hxml`, then other root HXML files,
then searches up to two directories below the root (excluding hidden directories,
dependency directories and common output directories). At each depth it prefers
`build.hxml`, then alphabetical order. Ambiguous choices are logged; builds and
targets are never merged. `extraParams.hxml` is not a build candidate.

The workspace's `haxelib.json`, and the nearest manifest owning each configured
class path, supply additional class paths and dependencies. Explicit HXML
library versions take precedence. Resolved dependencies contribute their
`extraParams.hxml` initialization macros. Class-path-only libraries do not
activate their own extra-parameter hooks. Without a selected HXML, the server
also reads `rayzor.toml` class paths and a workspace library's `extraParams.hxml`.
Missing libraries or invalid manifests are logged without discarding the other
settings. Later HXML class paths take precedence over earlier ones. Define
values remain available to macros through `Context.definedValue`.
Macro evaluation is native to Rayzor.

A library identified by a configured source path also satisfies inferred
dependencies on that library, avoiding an additional installed copy. A direct
HXML `-lib` still explicitly requests its library settings and hooks.

Unpinned libraries also honor checkouts registered with `haxelib dev`; an
explicit version still selects that installed version. Rayzor additionally
accepts local dependencies in `haxelib.json` as `{"path":"../library"}` or a
relative/absolute path string:

```json
{"dependencies":{"my_macros":{"path":"../my_macros"}}}
```

Paths are relative to the manifest that declares them. These libraries supply
their `classPath`, transitive dependencies and `extraParams.hxml` macros without
requiring installation in a haxelib repository. A local path can supply an
unpinned HXML `-lib` of the same name; an explicit HXML version remains
authoritative. Path-valued dependency declarations are a Rayzor extension to
the usual haxelib version strings.

An explicit `RAYZOR_STD_PATH` selects the analysis library. Otherwise the server
looks for the installed Haxe standard library, including `HAXE_STD_PATH` and
the installation behind `haxe` on `PATH`. The bundled Rayzor library is a
fallback when no Haxe installation is found.

Hover popovers include declaration comments, including imported declarations.
HaxeDoc (`/** ... **/`) and JSDoc (`/** ... */`) blocks preserve Markdown and
format `@param`, `@return`/`@returns`, `@throws`, `@see`, `@deprecated` and
`@example` tags. The same documentation is available in completion and
signature help. Adjacent line comments are also displayed.

Successful macro expansions are available as inlay hints with full expansion
tooltips. Hovering a macro invocation shows its expanded Haxe source. Macro
diagnostics retain their source file and can include a link to the invocation.

Clients can request a preview using `rayzor/expandMacro` with standard
`TextDocumentPositionParams`:

```json
{"textDocument":{"uri":"file:///project/Main.hx"},"position":{"line":4,"character":12}}
```

The response is `null` when no recorded expansion covers that position, or an
object with `name`, `expansion`, `range`, and an optional `definition` location.
Positions use zero-based UTF-16 offsets. Support is advertised as
`capabilities.experimental.macroExpansionProvider`. A client can display
`expansion` in a read-only Haxe buffer without executing the macro again.

---

## `rayzor.toml`

```toml
[project]
name = "my-app"
entry = "src/Main.hx"

[build]
class-paths = ["src"]
preset = "application"
output = "build/my-app"

[cache]
enabled = true
```

Workspace:

```toml
[workspace]
members = ["game", "engine", "tools/level-editor"]
```

Delegate to an existing HXML build instead of porting it:

```toml
[project]
name = "legacy-app"
hxml = "build.hxml"
```

Using native plugins? Declare **both** the class paths and the native libraries.
With only one, consumers fail to resolve at run time.

---

Internals: [Architecture](architecture/ARCHITECTURE.md).
