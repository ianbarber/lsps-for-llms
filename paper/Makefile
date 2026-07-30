# paper/Makefile — turn ../REPORT.md into an arXiv-style PDF.
#
# Pipeline:  pandoc (Markdown -> LaTeX)  ->  tectonic (LaTeX -> PDF)
#
# Targets:
#   make            regenerate report.tex from ../REPORT.md, then build report.pdf
#   make tex        regenerate report.tex only (needs pandoc)
#   make pdf        build report.pdf from the committed report.tex (tectonic only)
#   make clean      remove build byproducts (keeps report.tex and report.pdf)
#   make distclean  also remove the generated report.tex/report.pdf and bib copy
#
# report.tex is committed, so a reviewer WITHOUT pandoc can still run
# `make pdf` (or `tectonic report.tex`) with only tectonic installed.

REPORT   := ../REPORT.md
BIB_SRC  := ../docs/bibliography_efficiency.bib
BIB      := references.bib          # local copy so tectonic-alone builds work
TEX      := report.tex
PDF      := report.pdf

PANDOC   ?= pandoc
TECTONIC ?= tectonic

# --columns=72 (pandoc default) also drives table-width detection: wide tables
# get proportional wrapping p-columns, so they fit \textwidth.
#
# +lists_without_preceding_blankline matches GitHub. Plain `markdown` needs a
# blank line above a list; GitHub does not, so a list written without one
# renders on GitHub and comes out as literal asterisks in the PDF.
PANDOC_FLAGS := \
  --standalone \
  --from=markdown+lists_without_preceding_blankline \
  --to=latex \
  --no-highlight \
  --columns=72 \
  --number-sections \
  --lua-filter=report-filters.lua \
  --include-in-header=preamble.tex \
  --include-after-body=references.tex \
  --metadata-file=metadata.yaml

.PHONY: all tex pdf clean distclean
all: $(PDF)

# Keep a local copy of the bibliography next to report.tex.
$(BIB): $(BIB_SRC)
	cp $(BIB_SRC) $(BIB)

# Markdown -> LaTeX
$(TEX): $(REPORT) preamble.tex references.tex report-filters.lua metadata.yaml
	$(PANDOC) $(PANDOC_FLAGS) --output=$(TEX) $(REPORT)

tex: $(TEX)

# LaTeX -> PDF. tectonic auto-fetches only the packages it needs.
$(PDF): $(TEX) arxiv.sty $(BIB)
	$(TECTONIC) --keep-logs $(TEX)

pdf: $(PDF)

clean:
	rm -f report.aux report.log report.out report.bbl report.blg report.toc report.bcf report.run.xml

distclean: clean
	rm -f $(TEX) $(PDF) $(BIB)
