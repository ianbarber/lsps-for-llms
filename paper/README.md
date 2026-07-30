# paper/ — arXiv-style PDF build for REPORT.md

Turns the project's top-level `REPORT.md` into an arXiv-preprint PDF
(`report.pdf`) in one command. `REPORT.md` is the source of truth; `report.tex`
and `report.pdf` are generated from it and committed, so both are stale the
moment the report changes. Run `make` after editing it.

## What's here

| File | Role |
|---|---|
| `Makefile` | One-command build: `pandoc` (Markdown→LaTeX) then `tectonic` (LaTeX→PDF). |
| `arxiv.sty` | Vendored arXiv single-column preprint style (George Kour, NeurIPS-based). Self-contained; no network needed for the style. |
| `report-filters.lua` | Pandoc Lua filter: title from H1, `## Abstract`→`\begin{abstract}`, rewrites in-repo links to GitHub URLs, and converts the links listed in `LINK_CITE` into real `\citep` citations. A link whose URL is not in that table stays a bare hyperlink in the PDF, so add new citations there as well as to the bibliography. |
| `preamble.tex` | LaTeX injected into pandoc's template: loads `arxiv.sty` + `natbib`, maps Unicode glyphs, shrinks/wraps wide tables. |
| `references.tex` | Emits the `References` section (`\nocite{*}` + `natbib`). |
| `metadata.yaml` | Author, date, link colors. (Title is auto-taken from the report H1.) |
| `report.tex` | **Committed** generated LaTeX — lets a reviewer without pandoc build with tectonic alone. |
| `report.pdf` | **Committed** generated PDF — the preprint linked from the top-level README. |
| `references.bib` | Build-time copy of `../docs/bibliography_efficiency.bib` (canonical source stays in `docs/`). |

## Prerequisites

Two single-binary tools on your `PATH` (both install to `~/.local/bin`):

- **pandoc** — download the `linux-arm64` (aarch64) static tarball from
  <https://github.com/jgm/pandoc/releases>, extract, copy `bin/pandoc` to `~/.local/bin`.
- **tectonic** — download the `aarch64-unknown-linux-musl` tarball from
  <https://github.com/tectonic-typesetting/tectonic/releases>, copy `tectonic` to
  `~/.local/bin`. (Or use the official installer: `curl -fsSL https://drop-sh.fullyjustified.net | sh`.)
  Do **not** install full TeX Live — tectonic auto-fetches only the packages it needs
  (first build downloads them; later builds are cached).

Verify: `pandoc --version` and `tectonic --version`.

## Build

```bash
cd paper
make          # ../REPORT.md -> report.tex -> report.pdf
```

Output: `report.pdf`.

## Regenerate after REPORT.md changes

Just run `make` again — the `.tex` and `.pdf` rebuild from the current
`../REPORT.md`. Targets:

```bash
make          # regenerate report.tex from ../REPORT.md, then build report.pdf
make tex      # regenerate report.tex only (needs pandoc)
make pdf      # build report.pdf from the committed report.tex (tectonic ONLY, no pandoc)
make clean    # remove build byproducts (keeps report.tex / report.pdf)
make distclean# also remove generated report.tex / report.pdf / references.bib
```

`make pdf` is the "no pandoc" path: with only tectonic installed you can compile
the committed `report.tex`.

## How the report's quirks are handled

- **Title / abstract.** `report-filters.lua` uses the report's H1 as the title
  and moves the `## Abstract` section into a real `\begin{abstract}`.
- **Claim-ledger links.** In-repo relative links such as
  `[C6](evidence/claim_ledger.md#c6)` are rewritten to hyperlinks against
  `https://github.com/ianbarber/lsps-for-llms/blob/main/…`. External links
  (arXiv, etc.) are left as-is. (Chosen over plain-text so the references stay clickable.)
- **Wide tables.** pandoc emits proportional wrapping `p{}` columns for wide
  tables; `preamble.tex` adds `\small` + tighter `\tabcolsep` so every table
  (practitioner guide, checker grid, related work) fits the text width. Verified:
  zero `Overfull \hbox` warnings.
- **Unicode.** tectonic runs XeTeX with the Times text font, which lacks a few
  glyphs; `preamble.tex` maps `→ — – … −` to LaTeX equivalents (zero
  "Missing character" warnings).
- **Bibliography.** `docs/bibliography_efficiency.bib` is wired in via `natbib`.
  The current draft cites related work with inline hyperlinks, so `references.tex`
  uses `\nocite{*}` to list all entries. **To switch to real citations:** in
  `REPORT.md` replace an inline link, e.g. `[Typed Holes](https://arxiv.org/abs/2409.00921)`,
  with plain text, cite it with `\citep{blinn2024typedholes}` (keys are in the
  `.bib`), and drop `\nocite{*}` from `references.tex`.

## Known manual step

None for the build itself. For a **camera-ready** pass you would: convert the
inline related-work hyperlinks to `\citep{…}` and remove `\nocite{*}`; verify the
DAgger (2011) and other flagged `.bib` entries (see the notes in the `.bib`).
The author block in `metadata.yaml` carries no affiliation, which is deliberate.
