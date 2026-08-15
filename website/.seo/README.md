# Metadata the site must keep

`legacy-head.html` is the `<head>` of the site as it stood before the redesign,
captured verbatim. A redesign replaces the markup a page is built from, and
metadata is the part with no visible symptom when it goes missing — the page
looks finished while its search result and social card quietly regress.

Diff every new page's `<head>` against this file before shipping. Do not trust
that a copy was complete.

## Carries over unchanged

The canonical host is `https://rayzor-blade.com/`, and the home page must keep
that exact URL — new pages are additions beneath it, never a relocation of the
existing one.

- `robots` (`index, follow, max-image-preview:large`)
- `theme-color` `#f97316`
- icons: `/favicon.svg`, `/favicon.png` (32x32), `/apple-touch-icon.png`
- Open Graph: `og:url`, `og:type` (website), `og:site_name` (Rayzor),
  `og:title`, `og:description`, `og:image`
  (`https://rayzor-blade.com/ograph-image.png`, 1200x630, `image/png`)
- Twitter: `summary_large_image`, `twitter:domain`, `twitter:url`,
  `twitter:title`, `twitter:description`, `twitter:image` (+ dimensions)
- two JSON-LD blocks: `SoftwareApplication` and `Organization`

`ograph-image.png` is referenced by absolute URL from both the Open Graph and
Twitter tags, so its path must not move.

## Deliberately changed in the redesign

The title and description were rewritten for search intent. The old pair —
`Rayzor — High Performance Haxe Compiler` and "A next-generation Haxe compiler
with 5-tier JIT, ownership-based memory, and LLVM-powered native code
generation. Built with Rust." — described the implementation. Backend names are
not what anyone types into a search box, so they moved to the structured
`featureList` and the new pair leads with what someone is looking for:

- `<title>` — `Rayzor — Faster Haxe, Native Performance`
- tagline — "Faster Haxe, Native Performance", used for `og:description` and
  `twitter:description`, where a card truncates anything longer

The `keywords` term list and the JSON-LD `featureList` are where the technical
vocabulary lives now. Note that search engines rank on neither: `keywords` is
ignored outright, and `featureList` matters because it can reach a rich result,
not because it is a ranking signal. The title, the description and the page's
own headings are what actually carry weight.

## Needs per-page values, not copies

The redesign splits one page into five — Home, Architecture, CLI, Concurrency,
Docs. Reusing the home page's tags on all of them makes them read as duplicates
to a crawler, which is worse than having none.

Each page needs its own `<title>`, `description`, `canonical`, `og:title`,
`og:description`, `twitter:title`, `twitter:description`, and `og:url` /
`twitter:url`. The site-level values — `og:site_name`, `og:image`, icons,
`theme-color`, `robots` — stay identical across all five.

Keep the JSON-LD blocks on the home page. `SoftwareApplication` describes the
product as a whole and repeating it per page misrepresents the site as several
products.

## Content, not just metadata

`uploads/Rayzor---High-Performance-Haxe-Compiler-08-15-2026_08_30_PM.png` in the
design project is a screenshot of the OLD site, kept as reference. It records
what the page said, so it is the check for whether the restructure dropped copy
or a section — not a preview of the new design. The `.dc.html` files are the
only source for what the new pages should look like.
