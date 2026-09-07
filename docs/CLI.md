# Rayzor CLI

`rayzor <command> --help` has the full flag list. This page is the short path:
install, then the commands you use daily.

Compiler diagnostics — `dump`, `debug`, stage inspection, the `RAYZOR_*`
variables — are in [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Install

```bash
curl -fsSL https://rayzor.tech/install.sh | sh     # macOS, Linux, WSL
```

```powershell
irm https://rayzor.tech/install.ps1 | iex          # Windows
```

Lands in `~/.rayzor/bin` — add it to your `PATH` if the installer says so. The
download is self-contained: nothing else to install, no LLVM on the side.

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
rayzor aot main.hx --emit llvm-ir -o app.ll   # exe | obj | llvm-ir | llvm-bc | asm
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
| `script` | one-shot scripts — instant startup, no promotion |
| `application` | apps and servers — balanced, includes LLVM (**default**) |
| `server` | long-running services — aggressive optimization |
| `benchmark` | performance testing — immediate bailout |
| `development` | debugging — verbose logging |
| `embedded` | constrained targets — interpreter only |

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

Using native plugins? Declare **both** the class paths and the native libraries —
with only one, consumers fail to resolve at run time.

---

Internals: [Architecture](architecture/ARCHITECTURE.md).
