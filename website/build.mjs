#!/usr/bin/env node
// Renders the Design Composer sources in _import/ into the static pages the
// site ships.
//
// The imported .dc.html files are vendored verbatim — they are the design's
// source of truth and are never edited. Every transformation lives here:
// template directives are expanded at build time so each page is real HTML a
// crawler can read, and the component logic is carried over unchanged to drive
// the interactive parts in the browser.
//
//   node website/build.mjs [--results <dir>] [--out <dir>]
//
// --results points at a directory of results_<date>.json files; the newest one
// supplies the compiler benchmark chart. Defaults to compiler/benchmarks/results.

import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const IMPORT_DIR = path.join(HERE, "_import");

const argv = process.argv.slice(2);
const argOf = (flag, fallback) => {
  const i = argv.indexOf(flag);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};
// Naming a results directory is a claim that it holds the run being published.
// If it turns out to be empty the build stops, rather than quietly shipping
// whatever figures were last committed under a caption naming another machine.
const RESULTS_ARG = argOf("--results", null);
const RESULTS_DIR = path.resolve(RESULTS_ARG || path.join(ROOT, "compiler/benchmarks/results"));
const OUT_DIR = path.resolve(argOf("--out", HERE));

// ---------------------------------------------------------------------------
// Site metadata
//
// The values here are the ones the pre-redesign site was indexed under. They
// are recorded in .seo/legacy-head.html; changing one changes how the site
// appears in a search result or a shared link, so treat them as data, not copy.
// ---------------------------------------------------------------------------

const SITE = {
  origin: "https://rayzor.tech",
  name: "Rayzor",
  themeColor: "#f97316",
  image: "https://rayzor.tech/ograph-image.png",
  // Built from terms someone would type into a search box, not from the
  // implementation's vocabulary — backend names belong in featureList.
  tagline: "Faster Haxe, Native Performance",
  description:
    "Faster Haxe with native performance: a tiered JIT that turns Haxe into native machine code, with WebAssembly output, ownership memory and no garbage collector.",
  keywords:
    "Haxe, Haxe runtime, Haxe compiler, native Haxe, fast Haxe, JIT compilation, tiered JIT, WebAssembly, AOT compiler, ownership memory, no garbage collector, SIMD",
  features: [
    "Tiered JIT: interpreter, Cranelift and LLVM, promoted per function",
    "Ahead-of-time native binaries through LLVM",
    "WebAssembly core modules and WASI P2 components",
    "Ownership-based memory management with no garbage collector",
    "SIMD types with operator overloading",
    "Real OS threads with channels, select and worker pools",
    "Incremental module caching and portable bundles",
  ],
};

const PAGES = [
  {
    src: "Rayzor Home.dc.html",
    out: "index.html",
    canonical: "/",
    title: "Rayzor — Faster Haxe, Native Performance",
    description: SITE.description,
    social: SITE.tagline,
    jsonLd: true,
    interactive: true,
    buildState: { barsIn: true },
  },
  {
    src: "Rayzor Docs.dc.html",
    out: "docs.html",
    canonical: "/docs.html",
    title: "Rayzor Docs — Compilation modes, tiering and memory",
    description:
      "Rayzor compiles Haxe 4.x to native code. Pick a compilation mode, understand tier promotion, and read the memory model, artifacts and debug toolkit.",
  },
  {
    src: "Rayzor Architecture.dc.html",
    out: "architecture.html",
    canonical: "/architecture.html",
    title: "Rayzor Architecture — Pipeline, SSA and backends",
    description:
      "How Rayzor lowers Haxe to machine code: a shared SSA pipeline optimized once, a five-level tiered runtime, SIMD types, and the Cranelift, LLVM and WASM backends.",
    interactive: true,
  },
  {
    src: "Rayzor CLI.dc.html",
    out: "cli.html",
    canonical: "/cli.html",
    title: "Rayzor CLI — Commands, flags and environment",
    description:
      "Every Rayzor command, flag and environment variable: run, build, aot, bundle and the debug toolkit, with the optimization presets each one accepts.",
  },
  {
    src: "Rayzor Concurrency.dc.html",
    out: "concurrency.html",
    canonical: "/concurrency.html",
    title: "Rayzor Concurrency — Threads, channels and worker pools",
    description:
      "Real OS threads with channels, select, mutexes and atomics — with the sharing rules checked at compile time, plus NUMA-aware worker pools and spin pools.",
  },
];

const PAGE_LINKS = new Map(PAGES.map((p) => [p.src, p.out === "index.html" ? "/" : "/" + p.out]));

// ---------------------------------------------------------------------------
// HTML tokenizer
// ---------------------------------------------------------------------------

const VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input",
  "link", "meta", "source", "track", "wbr",
]);

function tokenize(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const lt = src.indexOf("<", i);
    if (lt < 0) {
      out.push({ t: "text", v: src.slice(i) });
      break;
    }
    if (lt > i) out.push({ t: "text", v: src.slice(i, lt) });

    if (src.startsWith("<!--", lt)) {
      const end = src.indexOf("-->", lt);
      const stop = end < 0 ? src.length : end + 3;
      out.push({ t: "comment", v: src.slice(lt, stop) });
      i = stop;
      continue;
    }
    if (src.startsWith("<!", lt)) {
      const end = src.indexOf(">", lt);
      const stop = end < 0 ? src.length : end + 1;
      out.push({ t: "text", v: src.slice(lt, stop) });
      i = stop;
      continue;
    }
    if (src[lt + 1] === "/") {
      const end = src.indexOf(">", lt);
      out.push({ t: "close", name: src.slice(lt + 2, end).trim().toLowerCase() });
      i = end + 1;
      continue;
    }

    let j = lt + 1;
    while (j < src.length && !/[\s/>]/.test(src[j])) j++;
    const name = src.slice(lt + 1, j).toLowerCase();
    const attrs = [];
    let selfClose = false;

    while (j < src.length) {
      while (j < src.length && /\s/.test(src[j])) j++;
      if (src[j] === ">") { j++; break; }
      if (src[j] === "/" && src[j + 1] === ">") { selfClose = true; j += 2; break; }

      let k = j;
      while (k < src.length && !/[\s=/>]/.test(src[k])) k++;
      const attrName = src.slice(j, k);
      let attrValue = null;
      let m = k;
      while (m < src.length && /\s/.test(src[m])) m++;
      if (src[m] === "=") {
        m++;
        while (m < src.length && /\s/.test(src[m])) m++;
        const quote = src[m];
        if (quote === '"' || quote === "'") {
          const end = src.indexOf(quote, m + 1);
          attrValue = src.slice(m + 1, end < 0 ? src.length : end);
          m = (end < 0 ? src.length : end) + 1;
        } else {
          let e = m;
          while (e < src.length && !/[\s>]/.test(src[e])) e++;
          attrValue = src.slice(m, e);
          m = e;
        }
        j = m;
      } else {
        j = k;
      }
      if (attrName) attrs.push([attrName, attrValue]);
    }

    out.push({ t: "open", name, attrs, selfClose: selfClose || VOID_TAGS.has(name) });
    i = j;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Template compiler
//
// Compiles the design's markup into a JS function body that appends HTML to an
// output array. The same compiled function runs twice: once here to produce the
// static page, and once in the browser to re-render after a state change. One
// implementation means the two can not drift.
// ---------------------------------------------------------------------------

const LITERAL = /^(true|false|null|-?\d+(\.\d+)?|"[^"]*"|'[^']*')$/;

function compileExpr(source, scope) {
  const text = source.trim();
  if (!text) throw new Error("empty binding");
  if (LITERAL.test(text)) return text;
  const root = text.split(/[.[(]/)[0].trim();
  if (!/^[A-Za-z_$][\w$]*$/.test(root)) throw new Error(`unsupported binding: {{ ${text} }}`);
  return scope.includes(root) ? text : "V." + text;
}

function hasBinding(value) {
  return value != null && value.includes("{{");
}

// Splits a string on {{ }} and returns a JS expression producing the result.
// `wrap` names the escaping helper applied to each interpolated value.
function compileInterp(value, scope, wrap) {
  const parts = [];
  const re = /\{\{([\s\S]*?)\}\}/g;
  let i = 0;
  let m;
  while ((m = re.exec(value))) {
    if (m.index > i) parts.push(JSON.stringify(value.slice(i, m.index)));
    parts.push(`${wrap}(${compileExpr(m[1], scope)})`);
    i = m.index + m[0].length;
  }
  if (i < value.length) parts.push(JSON.stringify(value.slice(i)));
  return parts.length ? parts.join("+") : '""';
}

// ---------------------------------------------------------------------------
// Responsive layout
//
// The design is authored at desktop width with inline styles, which no
// stylesheet can override without !important. Rather than rewrite the design's
// markup, each element's static style is read here and the handful of
// declarations that don't survive a narrow viewport — multi-column grids, the
// display sizes, page gutters, the header bar — get a generated class carrying
// the narrow-viewport value. Elements whose style is a binding are left alone;
// none of them carry layout.
// ---------------------------------------------------------------------------

const BP = { wide: 1024, mid: 900, bar: 860, narrow: 760, phone: 680 };

function parseStyle(text) {
  const map = new Map();
  for (const part of text.split(";")) {
    const i = part.indexOf(":");
    if (i < 0) continue;
    map.set(part.slice(0, i).trim(), part.slice(i + 1).trim());
  }
  return map;
}

// Splits on top-level whitespace, so repeat(3,minmax(0,1fr)) stays one token.
function splitColumns(value) {
  const out = [];
  let depth = 0;
  let cur = "";
  for (const ch of value) {
    if (ch === "(") depth++;
    else if (ch === ")") depth--;
    if (/\s/.test(ch) && depth === 0) {
      if (cur) out.push(cur);
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function px(value) {
  const m = /^(-?\d+(?:\.\d+)?)px$/.exec(String(value).trim());
  return m ? Number(m[1]) : null;
}

// Where a grid should give up its columns. Label grids with a narrow first
// column (a bullet, a number, an icon) are layout that already fits, so they
// keep their shape at every width.
function gridBreakpoints(value) {
  const repeat = /^repeat\(\s*(\d+)\s*,([\s\S]*)\)$/.exec(value.trim());
  if (repeat) {
    const count = Number(repeat[1]);
    if (count >= 4) return [[BP.wide, "repeat(2,minmax(0,1fr))"], [BP.narrow, "1fr"]];
    if (count === 3) return [[BP.mid, "repeat(2,minmax(0,1fr))"], [BP.phone, "1fr"]];
    if (count === 2) return [[BP.narrow, "1fr"]];
    return null;
  }
  const columns = splitColumns(value);
  if (columns.length < 2) return null;
  const fixed = columns.reduce((sum, c) => sum + (px(c) || 0), 0);
  if (fixed > 0 && fixed < 120) return null;
  // Sidebars and three-part layouts run out of room before an even split does.
  return [[fixed >= 120 || columns.length >= 3 ? BP.wide : BP.mid, "1fr"]];
}

// A display size scales with the viewport and stops at a floor that still
// reads as a heading on a phone.
function clampFont(size) {
  const floor = Math.min(40, Math.round(size * 0.62));
  const vw = Math.round((size / 12) * 100) / 100;
  return `clamp(${floor}px, ${vw}vw, ${size}px)`;
}

function shorthand(value) {
  const parts = value.trim().split(/\s+/);
  if (parts.length === 1) return [parts[0], parts[0], parts[0], parts[0]];
  if (parts.length === 2) return [parts[0], parts[1], parts[0], parts[1]];
  if (parts.length === 3) return [parts[0], parts[1], parts[2], parts[1]];
  if (parts.length === 4) return parts;
  return null;
}

function responsiveRules(tag, styleText, collapsedParentAt) {
  const style = parseStyle(styleText);
  const rules = []; // [maxWidth, declarations]
  const add = (width, declaration) => rules.push([width, declaration]);
  let collapsesAt = null;

  const columns = style.get("grid-template-columns");
  if (columns && (style.get("display") || "").includes("grid")) {
    for (const [width, value] of gridBreakpoints(columns) || []) {
      add(width, `grid-template-columns:${value}`);
      if (value === "1fr") collapsesAt = width;
    }
  }

  // A grid of uppercase labels is a table's column header. Stacked, it is three
  // captions for columns that no longer exist, so it goes with them.
  if (collapsesAt && style.get("text-transform") === "uppercase") {
    add(collapsesAt, "display:none");
  }

  // A column divider drawn as a side border turns into a stray vertical line
  // once the columns are gone; the same rule reads correctly along the bottom.
  if (collapsedParentAt) {
    for (const side of ["border-right", "border-left"]) {
      const value = style.get(side);
      if (value && !style.has("border-bottom")) {
        add(collapsedParentAt, `${side}-width:0;border-bottom:${value}`);
      }
    }
    // Uneven side padding is a column's inset from its neighbour. Stacked, it
    // reads as cells that fail to line up.
    const cell = style.get("padding") && shorthand(style.get("padding"));
    if (cell && cell[1] !== cell[3]) add(collapsedParentAt, "padding-left:0;padding-right:0");
  }

  const gap = px((style.get("gap") || "").split(/\s+/)[0]);
  if (gap != null && gap >= 40) {
    add(BP.wide, `gap:${Math.round(gap * 0.55)}px`);
    add(BP.phone, `gap:${Math.min(24, Math.round(gap * 0.4))}px`);
  }

  const fontSize = px(style.get("font-size"));
  if (fontSize != null && fontSize >= 30) add(null, `font-size:${clampFont(fontSize)}`);

  const padding = style.get("padding") && shorthand(style.get("padding"));
  if (padding) {
    const [t, r, b, l] = padding.map((v) => (px(v) == null ? v : px(v)));
    const gutter = (v) => (typeof v === "number" && v >= 26 ? 18 : v);
    const band = (v) => (typeof v === "number" && v >= 80 ? Math.round(v * 0.6) : v);
    const next = [band(t), gutter(r), band(b), gutter(l)];
    if (next.some((v, i) => v !== [t, r, b, l][i])) {
      add(BP.phone, `padding:${next.map((v) => (typeof v === "number" ? v + "px" : v)).join(" ")}`);
    }
  }

  if (style.get("position") === "sticky" && tag === "aside") add(BP.wide, "position:static");

  const isFlexRow =
    (style.get("display") || "").includes("flex") &&
    !style.has("flex-wrap") &&
    !(style.get("flex-direction") || "").startsWith("column");

  // A fixed-height flex row is a bar. Narrow, it has to grow to as many rows
  // as its contents need instead of pushing them past the edge.
  const barHeight = px(style.get("height"));
  if (isFlexRow && barHeight != null && barHeight >= 48) {
    add(
      BP.bar,
      `height:auto;min-height:${barHeight}px;flex-wrap:wrap;row-gap:10px;column-gap:14px;padding-top:10px;padding-bottom:10px`,
    );
  } else if (isFlexRow && !style.has("height") && ((gap != null && gap >= 20) || style.has("justify-content"))) {
    // Rows of separate items — spaced apart, or distributed across the line.
    // A tight unjustified gap is a label beside its value, which reads better
    // shrinking than breaking onto two lines.
    add(BP.phone, "flex-wrap:wrap");
  }

  // Ordered last so the bar keeps identity and actions together on the first
  // row and the links wrap underneath. Wrapping rather than scrolling: every
  // destination stays visible instead of hiding behind a gesture.
  if (tag === "nav") add(BP.bar, "order:10;width:100%;flex-wrap:wrap;gap:8px 18px");

  const width = px(style.get("width"));
  if (width != null && width >= 180) {
    if (tag === "input" || tag === "select" || tag === "textarea") {
      add(BP.bar, "width:auto;min-width:0;flex:1 1 140px");
    } else {
      add(BP.phone, "width:100%;max-width:100%");
    }
  }

  return { rules, collapsesAt };
}

// ---------------------------------------------------------------------------
// Mobile navigation
//
// A sticky header that wraps to three rows costs a third of a phone screen on
// every page. The links move behind a toggle instead — a checkbox and a label,
// so navigation works with scripting off and on pages that ship no script at
// all. The design's own markup is untouched: the toggle is inserted here, and
// the nav it reveals is the nav the design already wrote.
// ---------------------------------------------------------------------------

const MENU_ID = "rz-menu";

const BURGER = `<label class="rz-burger" for="${MENU_ID}" role="button" tabindex="0" aria-label="Navigation menu"><svg viewBox="0 0 20 20" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M3 6h14M3 10h14M3 14h14"></path></svg></label>`;

const MENU_TOGGLE = `<input type="checkbox" id="${MENU_ID}" class="rz-menu-state" aria-hidden="true" tabindex="-1">`;

// Marks the header links and the page's own sidebar as the panel the toggle
// reveals, and the secondary sidebar as the one a narrow screen drops.
function tagNavigation(tokens) {
  let asides = 0;
  let inHeader = false;
  for (const token of tokens) {
    if (token.t === "open" && token.name === "header") inHeader = true;
    if (token.t === "close" && token.name === "header") inHeader = false;
    if (token.t !== "open") continue;
    if (token.name === "nav" && inHeader) token.extraClass = "rz-nav";
    if (token.name === "aside") token.extraClass = asides++ === 0 ? "rz-nav" : "rz-toc";
    // Links out of the site and the search field move into the menu, so the bar
    // keeps room for the identity, the primary action and the toggle on one row.
    if (inHeader && token.name === "input") token.extraClass = "rz-menu-item";
    if (token.name === "a" && inHeader && /^https?:/.test(attr(token, "href") || "")) {
      token.extraClass = "rz-menu-item";
    }
  }
}

const DISCORD_URL = "https://discord.gg/NYdr8eWxF4";
// Tabler's "brand-discord" glyph, on a 24-unit grid rather than GitHub's 16.
const DISCORD_PATH =
  "M14.983 3l.123 .006c2.014 .214 3.527 .672 4.966 1.673a1 1 0 0 1 .371 .488c1.876 5.315 2.373 9.987 1.451 12.28c-1.003 2.005 -2.606 3.553 -4.394 3.553c-.732 0 -1.693 -.968 -2.328 -2.045a21.512 21.512 0 0 0 2.103 -.493a1 1 0 1 0 -.55 -1.924c-3.32 .95 -6.13 .95 -9.45 0a1 1 0 0 0 -.55 1.924c.717 .204 1.416 .37 2.103 .494c-.635 1.075 -1.596 2.044 -2.328 2.044c-1.788 0 -3.391 -1.548 -4.428 -3.629c-.888 -2.217 -.39 -6.89 1.485 -12.204a1 1 0 0 1 .371 -.488c1.439 -1.001 2.952 -1.459 4.966 -1.673a1 1 0 0 1 .935 .435l.063 .107l.651 1.285l.137 -.016a12.97 12.97 0 0 1 2.643 0l.134 .016l.65 -1.284a1 1 0 0 1 .754 -.54l.122 -.009zm-5.983 7a2 2 0 0 0 -1.977 1.697l-.018 .154l-.005 .149l.005 .15a2 2 0 1 0 1.995 -2.15zm6 0a2 2 0 0 0 -1.977 1.697l-.018 .154l-.005 .149l.005 .15a2 2 0 1 0 1.995 -2.15z";

function setAttr(token, name, value) {
  const want = name.toLowerCase();
  const found = token.attrs.find(([k]) => k.toLowerCase() === want);
  if (found) found[1] = value;
  else token.attrs.push([name, value]);
}

/** Adds the Discord link beside the header's outbound link.
 *
 * The design files are vendored, so the invite cannot be authored in them. The
 * link is cloned from the outbound one already in the bar rather than written
 * out here, so it inherits that anchor's styling -- including the hover class
 * this build generates -- and the two stay identical if the design moves. Only
 * the destination, the glyph and the label differ; the icon sits on a 24-unit
 * grid where GitHub's is 16, so the viewBox travels with the path.
 */
function injectDiscordLink(tokens) {
  const header = tokens.findIndex((t) => t.t === "open" && t.name === "header");
  if (header < 0) throw new Error("no <header> to hang the Discord link on");
  const headerEnd = tokens.findIndex((t, i) => i > header && t.t === "close" && t.name === "header");

  let start = -1;
  for (let i = header; i < headerEnd; i++) {
    const t = tokens[i];
    if (t.t === "open" && t.name === "a" && /^https?:/.test(attr(t, "href") || "")) { start = i; break; }
  }
  if (start < 0) throw new Error("no outbound link in <header> to model the Discord link on");

  let depth = 0;
  let end = -1;
  for (let i = start; i < headerEnd; i++) {
    if (tokens[i].t === "open" && tokens[i].name === "a" && !tokens[i].selfClose) depth++;
    if (tokens[i].t === "close" && tokens[i].name === "a") {
      depth--;
      if (depth === 0) { end = i; break; }
    }
  }
  if (end < 0) throw new Error("the header's outbound link is never closed");

  const clone = tokens.slice(start, end + 1).map((t) => ({
    ...t,
    attrs: t.attrs ? t.attrs.map(([k, v]) => [k, v]) : t.attrs,
  }));

  let labelled = false;
  for (const t of clone) {
    if (t.t === "open" && t.name === "a") setAttr(t, "href", DISCORD_URL);
    if (t.t === "open" && t.name === "svg") {
      setAttr(t, "viewBox", "0 0 24 24");
      setAttr(t, "fill", "currentColor");
      setAttr(t, "stroke", "none");
    }
    if (t.t === "open" && t.name === "path") setAttr(t, "d", DISCORD_PATH);
    if (t.t === "text" && t.v.trim()) {
      // Pages whose header links read "GitHub \u2197" keep that marker, so the new
      // link sits in the row as a sibling rather than as the odd one out.
      t.v = labelled
        ? ""
        : t.v.replace(/\S[\s\S]*\S|\S/, (m) => (m.includes("\u2197") ? "Discord \u2197" : "Discord"));
      labelled = true;
    }
  }
  if (!labelled) throw new Error("the header's outbound link carries no label to rename");

  tokens.splice(end + 1, 0, ...clone);
}

function injectMobileMenu(tokens) {
  injectDiscordLink(tokens);
  const header = tokens.findIndex((t) => t.t === "open" && t.name === "header");
  if (header < 0) throw new Error("no <header> to hang the navigation toggle on");
  const headerEnd = tokens.findIndex((t, i) => i > header && t.t === "close" && t.name === "header");
  if (headerEnd < 0) throw new Error("<header> is never closed");

  // The toggle belongs at the end of the bar, and the checkbox ahead of the
  // header so a sibling selector can reach everything it controls.
  let bar = headerEnd;
  while (bar > header && !(tokens[bar - 1].t === "close" && tokens[bar - 1].name === "div")) bar--;
  if (bar === header) throw new Error("<header> has no bar element to place the toggle in");

  const root = tokens.findIndex((t) => t.t === "open");
  tokens.splice(bar - 1, 0, { t: "raw", v: BURGER });
  tokens.splice(root + 1, 0, { t: "raw", v: MENU_TOGGLE });
  tagNavigation(tokens);
}

class TemplateCompiler {
  constructor() {
    this.lines = [];
    this.scope = [];
    this.pending = [];
    this.css = new Map(); // "media|pseudo|declarations" -> class name
    this.events = new Set();
    this.rootId = null;
    this.stack = [];
  }

  // The nearest enclosing element, looked at past any directive frames, and the
  // width at which its columns collapse — null when it is not a grid that does.
  collapsedParentAt() {
    for (let i = this.stack.length - 1; i >= 0; i--) {
      if (this.stack[i].kind === "tag") return this.stack[i].collapsesAt || null;
    }
    return null;
  }

  // Literal HTML accumulates so consecutive static chunks emit as one push.
  emitLiteral(text) {
    if (text) this.pending.push(JSON.stringify(text));
  }

  emitExpr(js) {
    this.pending.push(js);
  }

  flush() {
    if (this.pending.length) {
      this.lines.push(`O.push(${this.pending.join(",")});`);
      this.pending = [];
    }
  }

  emitCode(line) {
    this.flush();
    this.lines.push(line);
  }

  // Every generated rule overrides an inline style, so each declaration has to
  // carry !important — nothing else outranks a style attribute.
  className(declarations, pseudo, media) {
    const key = `${media || ""}|${pseudo || ""}|${declarations}`;
    let name = this.css.get(key);
    if (!name) {
      name = `dc${pseudo === "hover" ? "h" : pseudo === "focus" ? "f" : "r"}${this.css.size}`;
      this.css.set(key, name);
    }
    return name;
  }

  cssText() {
    const byMedia = new Map();
    for (const [key, name] of this.css) {
      const [media, pseudo, declarations] = splitKey(key);
      const important = declarations
        .split(";")
        .map((d) => d.trim())
        .filter(Boolean)
        .map((d) => (d.includes("!important") ? d : d + " !important"))
        .join(";");
      const rule = `.${name}${pseudo ? ":" + pseudo : ""}{${important}}`;
      if (!byMedia.has(media)) byMedia.set(media, []);
      byMedia.get(media).push(rule);
    }
    const out = [];
    for (const [media, rules] of [...byMedia].sort(sortMedia)) {
      if (!media) out.push(rules.join("\n"));
      else out.push(`@media (max-width:${media}px){\n${rules.map((r) => "  " + r).join("\n")}\n}`);
    }
    return out.join("\n");
  }

  openTag(token) {
    const { name, attrs } = token;
    const classes = token.extraClass ? [token.extraClass] : [];
    let out = "<" + name;
    const literalAttrs = [];

    for (const [attrName, rawValue] of attrs) {
      const value = rawValue == null ? null : rawValue;

      if (attrName === "style-hover" || attrName === "style-focus") {
        classes.push(this.className(value, attrName === "style-hover" ? "hover" : "focus", ""));
        continue;
      }
      if (attrName === "style" && !hasBinding(value)) {
        const responsive = responsiveRules(name, value, this.collapsedParentAt());
        token.collapsesAt = responsive.collapsesAt;
        for (const [media, declarations] of responsive.rules) {
          classes.push(this.className(declarations, "", media == null ? "" : String(media)));
        }
      }
      if (attrName === "data-screen-label" || attrName.startsWith("hint-")) continue;

      if (attrName === "ref") {
        const m = /^\{\{\s*([\w$]+)\s*\}\}$/.exec(value || "");
        if (!m) throw new Error(`ref must bind a plain name, got ${value}`);
        literalAttrs.push({ literal: ` data-dc-ref="${m[1]}"` });
        continue;
      }
      if (/^on[A-Z]/.test(attrName)) {
        const m = /^\{\{([\s\S]*)\}\}$/.exec(value || "");
        if (!m) throw new Error(`${attrName} must bind a handler, got ${value}`);
        const event = attrName.slice(2).toLowerCase();
        this.events.add(event);
        literalAttrs.push({
          prefix: ` data-dc-on="${event}" data-dc-h="`,
          expr: `$h(${compileExpr(m[1], this.scope)})`,
          suffix: '"',
        });
        continue;
      }

      if (value == null) {
        literalAttrs.push({ literal: ` ${attrName}` });
        continue;
      }
      if (attrName === "class") {
        classes.unshift(value);
        continue;
      }
      if (!hasBinding(value)) {
        literalAttrs.push({ literal: ` ${attrName}="${value}"` });
        continue;
      }
      literalAttrs.push({
        prefix: ` ${attrName}="`,
        expr: compileInterp(value, this.scope, "$a"),
        suffix: '"',
      });
    }

    this.emitLiteral(out);
    for (const a of literalAttrs) {
      if (a.literal != null) this.emitLiteral(a.literal);
      else {
        this.emitLiteral(a.prefix);
        this.emitExpr(a.expr);
        this.emitLiteral(a.suffix);
      }
    }
    if (classes.length) this.emitLiteral(` class="${classes.join(" ")}"`);
    this.emitLiteral(token.selfClose && VOID_TAGS.has(name) ? ">" : ">");
  }

  run(tokens) {
    const stack = this.stack;
    for (const token of tokens) {
      if (token.t === "text") {
        if (!hasBinding(token.v)) this.emitLiteral(token.v);
        else {
          const re = /\{\{([\s\S]*?)\}\}/g;
          let i = 0;
          let m;
          while ((m = re.exec(token.v))) {
            if (m.index > i) this.emitLiteral(token.v.slice(i, m.index));
            this.emitExpr(`$e(${compileExpr(m[1], this.scope)})`);
            i = m.index + m[0].length;
          }
          if (i < token.v.length) this.emitLiteral(token.v.slice(i));
        }
        continue;
      }
      if (token.t === "comment") continue;
      if (token.t === "raw") {
        this.emitLiteral(token.v);
        continue;
      }

      if (token.t === "open") {
        if (token.name === "sc-for") {
          const list = attr(token, "list");
          const as = attr(token, "as");
          if (!list || !as) throw new Error("sc-for needs list= and as=");
          const inner = /^\{\{([\s\S]*)\}\}$/.exec(list.trim());
          if (!inner) throw new Error(`sc-for list must be a binding, got ${list}`);
          this.emitCode(`$list(${compileExpr(inner[1], this.scope)}).forEach(function(${as}, $i){`);
          this.scope.push(as);
          stack.push({ kind: "for" });
          continue;
        }
        if (token.name === "sc-if") {
          const value = attr(token, "value");
          if (!value) throw new Error("sc-if needs value=");
          const inner = /^\{\{([\s\S]*)\}\}$/.exec(value.trim());
          if (!inner) throw new Error(`sc-if value must be a binding, got ${value}`);
          this.emitCode(`if (${compileExpr(inner[1], this.scope)}) {`);
          stack.push({ kind: "if" });
          continue;
        }
        // The first element is the mount point. Naming it in the template
        // rather than wrapping it keeps the rendered tree identical to the
        // static one, so a re-render can not disagree with what was served.
        if (this.rootId === null) {
          const existing = attr(token, "id");
          this.rootId = existing || "rz-root";
          if (!existing) token.attrs.unshift(["id", this.rootId]);
        }
        this.openTag(token);
        if (!token.selfClose) stack.push({ kind: "tag", name: token.name, collapsesAt: token.collapsesAt });
        continue;
      }

      if (token.t === "close") {
        const top = stack[stack.length - 1];
        if (token.name === "sc-for" || token.name === "sc-if") {
          if (!top || top.kind === "tag") throw new Error(`unbalanced </${token.name}>`);
          stack.pop();
          if (top.kind === "for") {
            this.scope.pop();
            this.emitCode("});");
          } else this.emitCode("}");
          continue;
        }
        if (VOID_TAGS.has(token.name)) continue;
        if (top && top.kind === "tag" && top.name === token.name) stack.pop();
        this.emitLiteral(`</${token.name}>`);
      }
    }
    this.flush();
    if (stack.length) throw new Error(`unclosed ${stack.map((s) => s.kind + ":" + (s.name || "")).join(", ")}`);
    return this.lines.join("\n");
  }
}

function splitKey(key) {
  const a = key.indexOf("|");
  const b = key.indexOf("|", a + 1);
  return [key.slice(0, a), key.slice(a + 1, b), key.slice(b + 1)];
}

// Widest media query first, so a narrower one can override it.
function sortMedia([a], [b]) {
  if (!a) return -1;
  if (!b) return 1;
  return Number(b) - Number(a);
}

function attr(token, name) {
  const found = token.attrs.find(([n]) => n === name);
  return found ? found[1] : null;
}

function escapeAttr(text) {
  return String(text).replace(/&/g, "&amp;").replace(/"/g, "&quot;");
}

// ---------------------------------------------------------------------------
// Runtime helpers, shared by the build and the browser
// ---------------------------------------------------------------------------

const RUNTIME_HELPERS = `
function $e(v){ return v==null ? "" : String(v).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function $a(v){ return v==null ? "" : String(v).replace(/&/g,"&amp;").replace(/"/g,"&quot;"); }
function $list(v){ return Array.isArray(v) ? v : []; }
`;

function makeRenderer(body) {
  const src = `${RUNTIME_HELPERS}
return function(V, H){
  var O = [];
  function $h(fn){ H.push(fn); return H.length - 1; }
${body}
  return O.join("");
};`;
  return new Function(src)();
}

// ---------------------------------------------------------------------------
// Design Composer source
// ---------------------------------------------------------------------------

function parseDesignFile(file) {
  const src = fs.readFileSync(file, "utf8");

  const helmet = between(src, "<helmet>", "</helmet>");
  const bodyStart = src.indexOf("</helmet>");
  const bodyEnd = src.indexOf("</x-dc>");
  if (helmet == null || bodyStart < 0 || bodyEnd < 0) {
    throw new Error(`${path.basename(file)}: expected <x-dc> with a <helmet> block`);
  }
  const template = src.slice(bodyStart + "</helmet>".length, bodyEnd).trim();

  const scriptOpen = src.indexOf("data-dc-script");
  if (scriptOpen < 0) throw new Error(`${path.basename(file)}: no component script`);
  const scriptBodyStart = src.indexOf(">", scriptOpen) + 1;
  const scriptBodyEnd = src.indexOf("</script>", scriptBodyStart);
  const script = src.slice(scriptBodyStart, scriptBodyEnd);

  // Anything the design pulls in through the page <head> rather than the helmet
  // (Architecture loads mermaid this way).
  const headScripts = [];
  for (const m of src.slice(0, src.indexOf("<x-dc>")).matchAll(/<script src="(https:[^"]+)"><\/script>/g)) {
    headScripts.push(m[1]);
  }

  return {
    helmet,
    template: rewriteDesignLinks(template),
    script: rewriteDesignLinks(script),
    headScripts,
  };
}

// The design files link to each other by source filename, in markup and in the
// nav data alike. One rewrite covers both.
function rewriteDesignLinks(text) {
  let out = text;
  for (const [src, target] of PAGE_LINKS) out = out.split(src).join(target);
  return out;
}

function between(src, open, close) {
  const a = src.indexOf(open);
  if (a < 0) return null;
  const b = src.indexOf(close, a);
  if (b < 0) return null;
  return src.slice(a + open.length, b);
}

// Evaluates the design's component to get the values its template binds to.
function evaluateComponent(script, buildState) {
  const context = vm.createContext({
    console,
    setTimeout: () => 0,
    clearTimeout: () => {},
    queueMicrotask: () => {},
    React: { createRef: () => ({ current: null }) },
    __out: null,
  });
  const preamble = `
class DCLogic {
  setState(patch){ Object.assign(this.state, typeof patch === "function" ? patch(this.state) : patch); }
}
`;
  const epilogue = `
;__out = (function(){
  var c = new Component();
  Object.assign(c.state || (c.state = {}), ${JSON.stringify(buildState || {})});
  return c.renderVals();
})();
`;
  vm.runInContext(preamble + script + epilogue, context, { filename: "component.js" });
  return context.__out;
}

// ---------------------------------------------------------------------------
// Benchmark data
//
// The compiler chart is CI output, not copy. The newest results file wins, and
// the hardware caption is read from the same file so the numbers and the
// machine they came from can never describe different runs.
// ---------------------------------------------------------------------------

// Display names, in the order the chart lists them. A target missing from a
// results file is simply absent from that chart.
const CHART_TARGETS = [
  ["rayzor-tiered", "Rayzor · tiered", true],
  ["rayzor-llvm", "Rayzor · LLVM", true],
  ["rayzor-cranelift", "Rayzor · Cranelift", true],
  ["haxe-cpp", "hxcpp", false],
  ["haxe-jvm", "Haxe/JVM", false],
  ["haxe-hashlink-c", "HashLink/C", false],
  ["haxe-hashlink", "HashLink", false],
];

const OS_LABELS = { macos: "macOS", linux: "linux", windows: "windows" };

function loadBenchmarks(dir) {
  if (!fs.existsSync(dir)) return null;
  const files = [];
  const walk = (d) => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (/^results_\d{4}-\d{2}-\d{2}\.json$/.test(entry.name)) files.push(full);
    }
  };
  walk(dir);
  if (!files.length) return null;
  files.sort((a, b) => path.basename(a).localeCompare(path.basename(b)));
  const newest = files[files.length - 1];
  const data = JSON.parse(fs.readFileSync(newest, "utf8"));

  const workloads = [];
  const runs = {};
  const skipped = [];
  const untracked = new Set();
  for (const bench of data.benchmarks || []) {
    const rows = [];
    for (const [target, label, isRayzor] of CHART_TARGETS) {
      const hit = (bench.results || []).find((r) => r.target === target);
      if (!hit) continue;
      const row = {
        name: label,
        comp: round(hit.compile_time_ms),
        exec: round(hit.runtime_ms),
      };
      if (isRayzor) row.rz = 1;
      rows.push(row);
    }
    for (const r of bench.results || []) {
      if (!CHART_TARGETS.some(([t]) => t === r.target)) untracked.add(r.target);
    }
    // A chart of one bar compares nothing.
    if (rows.length < 2) {
      skipped.push(`${bench.name} (${rows.length} of ${CHART_TARGETS.length} targets)`);
      continue;
    }
    workloads.push({ id: bench.name, name: bench.name });
    runs[bench.name] = rows;
  }
  // Coverage the chart drops is reported, never silently absent — a kernel
  // missing from the page should be traceable to the run that lacked it.
  if (skipped.length) console.warn(`  ! kernels not charted: ${skipped.join(", ")}`);
  if (untracked.size) console.warn(`  ! targets not in the chart: ${[...untracked].join(", ")}`);
  if (!workloads.length) return null;

  const info = data.system_info || {};
  const iterations = firstIterations(data);
  const machine = [info.cpu_model, OS_LABELS[info.os] || info.os, info.arch].filter(Boolean).join(", ");
  const note =
    (iterations ? `${iterations} measured iterations, mean reported. ` : "") +
    (machine ? `${machine}, ` : "") +
    `${data.date}.`;

  return { workloads, runs, note, source: path.basename(newest) };
}

function firstIterations(data) {
  for (const bench of data.benchmarks || []) {
    for (const r of bench.results || []) if (r.iterations) return r.iterations;
  }
  return null;
}

function round(ms) {
  return Math.round((Number(ms) || 0) * 100) / 100;
}

// The caption the design shipped with. Replacing it is how the chart's machine
// and date stay tied to the data actually rendered; if the design rewords it,
// this must fail loudly rather than leave a stale machine on the page.
const BENCH_NOTE_ANCHOR = "15 warmup iterations, 10 measured, mean reported. AMD EPYC, linux x86_64, 2026-08-15.";

function applyBenchmarks(page, parsed, bench) {
  if (!page.interactive || !parsed.script.includes("const RUNS")) return parsed;
  if (!bench) {
    if (RESULTS_ARG) throw new Error(`no usable benchmark results under ${RESULTS_DIR}`);
    console.warn(`  ! no benchmark results under ${RESULTS_DIR} — keeping the design's figures`);
    return parsed;
  }
  if (!parsed.template.includes(BENCH_NOTE_ANCHOR)) {
    throw new Error("benchmark caption not found in the design source — update BENCH_NOTE_ANCHOR");
  }

  const injected =
    `\nWORKLOADS.length = 0;\n` +
    `WORKLOADS.push.apply(WORKLOADS, ${JSON.stringify(bench.workloads)});\n` +
    `for (var $k in RUNS) delete RUNS[$k];\n` +
    `Object.assign(RUNS, ${JSON.stringify(bench.runs)});\n`;

  // The component's own state decides which workload is shown; keep it valid
  // when CI reports a different set of kernels than the design assumed.
  const script =
    parsed.script.replace(/class Component extends DCLogic \{/, injected + "class Component extends DCLogic {") +
    `\n;(function(){ var d = Component.prototype; var base = d.renderVals;
  d.renderVals = function(){
    if (!RUNS[this.state.work]) this.state.work = WORKLOADS[0].id;
    var v = base.call(this);
    v.benchNote = ${JSON.stringify(bench.note)};
    return v;
  };
})();\n`;

  return { ...parsed, script, template: parsed.template.replace(BENCH_NOTE_ANCHOR, "{{ benchNote }}") };
}

// ---------------------------------------------------------------------------
// Page assembly
// ---------------------------------------------------------------------------

function buildHead(page, helmet, headScripts) {
  const url = SITE.origin + (page.canonical === "/" ? "/" : page.canonical);
  const social = page.social || page.description;
  const tags = [
    '<meta charset="utf-8">',
    '<meta name="viewport" content="width=device-width, initial-scale=1">',
    `<title>${escapeText(page.title)}</title>`,
    `<meta name="description" content="${escapeAttr(page.description)}">`,
    `<meta name="keywords" content="${escapeAttr(SITE.keywords)}">`,
    '<meta name="robots" content="index, follow, max-image-preview:large">',
    `<meta name="theme-color" content="${SITE.themeColor}">`,
    `<link rel="canonical" href="${url}">`,
    '<link rel="icon" href="/favicon.svg" type="image/svg+xml">',
    '<link rel="icon" href="/favicon.png" type="image/png" sizes="32x32">',
    '<link rel="apple-touch-icon" href="/apple-touch-icon.png">',
    "",
    `<meta property="og:url" content="${url}">`,
    '<meta property="og:type" content="website">',
    `<meta property="og:site_name" content="${SITE.name}">`,
    `<meta property="og:title" content="${escapeAttr(page.title)}">`,
    `<meta property="og:description" content="${escapeAttr(social)}">`,
    `<meta property="og:image" content="${SITE.image}">`,
    '<meta property="og:image:width" content="1200">',
    '<meta property="og:image:height" content="630">',
    '<meta property="og:image:type" content="image/png">',
    "",
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta property="twitter:domain" content="rayzor.tech">',
    `<meta property="twitter:url" content="${url}">`,
    `<meta name="twitter:title" content="${escapeAttr(page.title)}">`,
    `<meta name="twitter:description" content="${escapeAttr(social)}">`,
    `<meta name="twitter:image" content="${SITE.image}">`,
    '<meta name="twitter:image:width" content="1200">',
    '<meta name="twitter:image:height" content="630">',
  ];

  if (page.jsonLd) tags.push("", jsonLd());
  for (const src of headScripts) tags.push(`<script src="${src}" defer></script>`);

  return tags.map((t) => (t ? "    " + t : "")).join("\n") + "\n" + indent(helmet.trim(), 4);
}

// `keywords` is what search engines read as a term list; `featureList` is the
// SoftwareApplication-specific field that describes what the thing does, and is
// the one that can reach a rich result.
function jsonLd() {
  const application = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: SITE.name,
    alternateName: `Rayzor — ${SITE.tagline}`,
    description: SITE.description,
    url: SITE.origin + "/",
    applicationCategory: "DeveloperApplication",
    applicationSubCategory: "Compiler",
    operatingSystem: "Windows, macOS, Linux",
    keywords: SITE.keywords,
    featureList: SITE.features,
    programmingLanguage: "Haxe",
    license: "https://opensource.org/licenses/Apache-2.0",
    offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    author: { "@type": "Organization", name: SITE.name, url: SITE.origin + "/" },
    downloadUrl: "https://github.com/rayzor-blade/rayzor",
    softwareRequirements: "Rust toolchain",
  };
  const organization = {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: SITE.name,
    url: SITE.origin + "/",
    logo: SITE.origin + "/favicon.svg",
    keywords: SITE.keywords,
    sameAs: ["https://github.com/rayzor-blade/rayzor"],
  };
  return [application, organization]
    .map((node) => `<script type="application/ld+json">\n${JSON.stringify(node, null, 2)}\n</script>`)
    .join("\n");
}

function escapeText(text) {
  return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function indent(text, spaces) {
  const pad = " ".repeat(spaces);
  return text
    .split("\n")
    .map((line) => (line.trim() ? pad + line : line))
    .join("\n");
}

// The browser half: the design's component logic, unchanged, driving the same
// compiled template. A state change re-renders and the result is morphed into
// the live DOM so CSS transitions on width and colour still run.
// Applies to every page regardless of what the design put inline: nothing may
// be wider than the screen, and code keeps its own scrollbar instead of
// stretching the page.
const BASE_CSS = `
html, body { max-width: 100%; overflow-x: hidden; }
img, svg { max-width: 100%; height: auto; }
pre, table { max-width: 100%; overflow-x: auto; }

/* Code set outside a <pre> has no scroller of its own; on a phone a long
   expression has to break rather than run off the edge. */
@media (max-width: 640px) {
  [style*="JetBrains Mono"] { overflow-wrap: anywhere; }
}

.rz-menu-state { position: absolute; width: 0; height: 0; opacity: 0; pointer-events: none; }
.rz-burger {
  display: none; align-items: center; justify-content: center; flex: none;
  width: 38px; height: 34px; border: 1px solid #22242A; border-radius: 9px;
  color: #C9CCD2; cursor: pointer;
}
.rz-burger:hover { border-color: #4A4E58; color: #E9EAEE; }
.rz-menu-state:checked ~ header .rz-burger { border-color: #F97316; color: #F97316; }

@media (max-width: ${BP.bar}px) {
  .rz-burger { display: inline-flex; }
  .rz-nav { display: none !important; }
  .rz-menu-state:checked ~ header nav.rz-nav {
    display: flex !important; order: 10; width: 100%; flex-direction: column;
    align-items: flex-start; gap: 0; margin-top: 4px; padding-top: 6px;
    border-top: 1px solid #1A1C21;
  }
  .rz-menu-state:checked ~ header nav.rz-nav > a { padding: 9px 0; width: 100%; }
  .rz-menu-state:checked ~ * aside.rz-nav { display: block !important; }
  aside.rz-toc { display: none !important; }
  .rz-menu-item { display: none !important; }
  .rz-menu-state:checked ~ header .rz-menu-item {
    display: inline-flex !important; order: 11; width: 100%; padding: 9px 0;
  }
  .rz-menu-state:checked ~ header input.rz-menu-item {
    display: block !important; order: 9; width: 100%; margin-top: 4px;
  }
}
`;

function clientScript(renderBody, script, events, rootId) {
  return `<script type="module">
${RUNTIME_HELPERS}
const $refs = new Map();
const React = { createRef(){ const r = { name: null, get current(){
  if (!r.name) throw new Error("dc: ref read before it was bound to a field");
  return document.querySelector('[data-dc-ref="' + r.name + '"]');
} }; return r; } };

let $pending = false;
class DCLogic {
  setState(patch){
    Object.assign(this.state, typeof patch === "function" ? patch(this.state) : patch);
    if ($pending) return;
    $pending = true;
    queueMicrotask(() => { $pending = false; paint(); });
  }
}

${script}

const $render = (function(){
  return function(V, H){
    var O = [];
    function $h(fn){ H.push(fn); return H.length - 1; }
${renderBody}
    return O.join("");
  };
})();

const root = document.getElementById(${JSON.stringify(rootId)});
let handlers = [];

function paint(){
  handlers = [];
  const html = $render(component.renderVals(), handlers);
  const scratch = document.createElement("template");
  scratch.innerHTML = html;
  morph(root, scratch.content.firstElementChild);
}

function morph(live, next){
  if (live.nodeType !== next.nodeType || live.nodeName !== next.nodeName) { live.replaceWith(next); return; }
  if (live.nodeType === Node.TEXT_NODE) { if (live.data !== next.data) live.data = next.data; return; }
  for (const a of Array.from(next.attributes)) {
    if (live.getAttribute(a.name) !== a.value) live.setAttribute(a.name, a.value);
  }
  for (const a of Array.from(live.attributes)) {
    if (!next.hasAttribute(a.name)) live.removeAttribute(a.name);
  }
  const liveKids = Array.from(live.childNodes);
  const nextKids = Array.from(next.childNodes);
  const shared = Math.min(liveKids.length, nextKids.length);
  for (let i = 0; i < shared; i++) morph(liveKids[i], nextKids[i]);
  for (let i = shared; i < nextKids.length; i++) live.appendChild(nextKids[i]);
  for (let i = liveKids.length - 1; i >= shared; i--) liveKids[i].remove();
}

const component = new Component();
for (const key of Object.keys(component)) {
  const value = component[key];
  if (value && typeof value === "object" && "name" in value && value.name === null && "current" in value) {
    value.name = key;
  }
}

for (const type of ${JSON.stringify(events)}) {
  root.addEventListener(type, (event) => {
    const el = event.target.closest('[data-dc-on="' + type + '"]');
    if (!el || !root.contains(el)) return;
    const fn = handlers[Number(el.getAttribute("data-dc-h"))];
    if (typeof fn !== "function") throw new Error("dc: no handler bound for " + el.outerHTML.slice(0, 80));
    fn(event);
  });
}

paint();
if (component.componentDidMount) component.componentDidMount();
</script>`;
}

// ---------------------------------------------------------------------------

function buildPage(page, bench) {
  const file = path.join(IMPORT_DIR, page.src);
  const parsed = applyBenchmarks(page, parseDesignFile(file), bench);

  const tokens = tokenize(parsed.template);
  injectMobileMenu(tokens);

  const compiler = new TemplateCompiler();
  const renderBody = compiler.run(tokens);
  const render = makeRenderer(renderBody);

  const values = evaluateComponent(parsed.script, page.buildState);
  const html = render(values, []);

  const css = compiler.cssText();
  const head = buildHead(page, parsed.helmet, parsed.headScripts);

  const parts = [
    "<!doctype html>",
    '<html lang="en">',
    "  <head>",
    head,
    `    <style>\n${indent(BASE_CSS.trim(), 6)}\n${indent(css, 6)}\n    </style>`,
    "  </head>",
    "  <body>",
    "    " + html,
    page.interactive
      ? indent(clientScript(renderBody, parsed.script, [...compiler.events], compiler.rootId), 4)
      : "",
    "  </body>",
    "</html>",
    "",
  ].filter((p) => p !== "");

  fs.writeFileSync(path.join(OUT_DIR, page.out), parts.join("\n"));
  return { bytes: Buffer.byteLength(parts.join("\n")), dynamic: page.interactive === true };
}

/** Sitemap, generated from PAGES so a new page cannot be added without one.
    lastmod comes from each source's mtime: a date that moves only when the
    page actually changes is worth more to a crawler than today's date on
    every URL, which is what a build-time stamp would give. */
function writeSitemap() {
  const urls = PAGES.map((page) => {
    const loc = SITE.origin + (page.canonical === "/" ? "/" : page.canonical);
    const src = path.join(IMPORT_DIR, page.src);
    let lastmod = null;
    try {
      lastmod = fs.statSync(src).mtime.toISOString().slice(0, 10);
    } catch {}
    // The home page is the entry point; the rest are siblings of equal weight.
    const priority = page.canonical === "/" ? "1.0" : "0.8";
    return [
      "  <url>",
      `    <loc>${loc}</loc>`,
      lastmod ? `    <lastmod>${lastmod}</lastmod>` : null,
      "    <changefreq>weekly</changefreq>",
      `    <priority>${priority}</priority>`,
      "  </url>",
    ]
      .filter(Boolean)
      .join("\n");
  });

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...urls,
    "</urlset>",
    "",
  ].join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "sitemap.xml"), xml);
  return PAGES.length;
}

/** robots.txt, naming the sitemap so a crawler finds it without being told. */
function writeRobots() {
  const body = [
    "User-agent: *",
    "Allow: /",
    "",
    `Sitemap: ${SITE.origin}/sitemap.xml`,
    "",
  ].join("\n");
  fs.writeFileSync(path.join(OUT_DIR, "robots.txt"), body);
}

function copyAssets() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const name of ["logo.svg", "favicon.svg"]) {
    const from = path.join(IMPORT_DIR, name);
    if (fs.existsSync(from)) fs.copyFileSync(from, path.join(OUT_DIR, name));
  }
}

function main() {
  const bench = loadBenchmarks(RESULTS_DIR);
  if (bench) {
    console.log(`benchmarks: ${bench.source} — ${bench.workloads.map((w) => w.id).join(", ")}`);
    console.log(`            ${bench.note}`);
  } else if (RESULTS_ARG) {
    throw new Error(`--results ${RESULTS_ARG} holds no usable benchmark results`);
  } else {
    console.log(`benchmarks: none found under ${RESULTS_DIR}`);
  }

  copyAssets();
  for (const page of PAGES) {
    const { bytes, dynamic } = buildPage(page, bench);
    console.log(`  ${page.out.padEnd(18)} ${String(bytes).padStart(7)} bytes${dynamic ? "  + client" : ""}`);
  }
  const mapped = writeSitemap();
  writeRobots();
  console.log(`  ${"sitemap.xml".padEnd(18)} ${String(mapped).padStart(7)} urls`);
  console.log(`  ${"robots.txt".padEnd(18)}`);
}

main();
