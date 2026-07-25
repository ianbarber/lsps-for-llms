-- report-filters.lua
-- Adapts REPORT.md for the arXiv LaTeX build WITHOUT editing the source file.
-- Three jobs:
--   1. Rewrite in-repo relative links (e.g. evidence/claim_ledger.md#c6) to
--      absolute GitHub blob URLs, so claim-ledger references become valid
--      hyperlinks in the PDF instead of broken relative paths. External
--      links (arXiv, https://...) are left untouched.
--   2. Use the document's first level-1 heading as the paper title and drop it
--      from the body (the title is typeset by \maketitle).
--   3. Move the "## Abstract" section into the `abstract` metadata field so it
--      renders inside the arxiv.sty \begin{abstract} environment.

local stringify = pandoc.utils.stringify

-- Base URL for in-repo files that are NOT part of the PDF.
local GH_BASE = "https://github.com/ianbarber/lsps-for-llms/blob/main/"

-- Related-work arXiv links -> bib keys. Each of these appears once in the
-- "Related work" table as [Name](arXiv URL). We keep the visible Name and
-- append a real numbered citation, so it renders as "Name [n]" tied to the
-- References list (rather than a bare hyperlink). Verified against the .bib
-- arXiv IDs.
local ARXIV_CITE = {
  ["https://arxiv.org/abs/2409.00921"] = "blinn2024typedholes",       -- Typed Holes
  ["https://arxiv.org/abs/2510.22210"] = "go2025lsprag",              -- LSPRAG
  ["https://arxiv.org/abs/2406.10018"] = "liu2024stallplus",          -- STALL+
  ["https://arxiv.org/abs/2403.16792"] = "bi2024cocogen",             -- CoCoGen
  ["https://arxiv.org/abs/2604.05407"] = "kim2026codestruct",         -- CodeStruct
  ["https://arxiv.org/abs/2203.05132"] = "wang2022compcoder",         -- CompCoder
  ["https://arxiv.org/abs/2504.09246"] = "mundler2025typeconstrained",-- type-constrained generation
  ["https://arxiv.org/abs/2510.22907"] = "zhang2025rlcsf",            -- RLCSF v2
}

-- 1) Link rewriting.
--    a. Mapped arXiv links become "visible name + \citep{key}".
--    b. Other external links (URI scheme or #fragment) are left as hyperlinks.
--    c. In-repo relative paths are rebased onto GitHub.
function Link(el)
  local t = el.target
  local key = ARXIV_CITE[t]
  if key then
    local out = {}
    for _, x in ipairs(el.content) do out[#out + 1] = x end   -- keep the work name
    out[#out + 1] = pandoc.RawInline("latex", "~\\citep{" .. key .. "}")
    return out
  end
  local is_scheme = t:match("^%a[%w+.%-]*:") ~= nil
  local is_fragment = t:match("^#") ~= nil
  if not is_scheme and not is_fragment then
    el.target = GH_BASE .. t
  end
  return el
end

-- 1d) Methods citation: attach \citep{ross2011dagger} at the single textual
--     anchor "A DAgger-style relabel run ..." in section 2. DAgger has no
--     arXiv URL, so we match the Str tokens "DAgger-style" then "relabel"
--     (the phrase occurs exactly once) and splice the cite right after
--     "relabel". REPORT.md itself is never edited.
local dagger_done = false
function Inlines(inlines)
  if dagger_done then return nil end
  local t = {}
  for _, x in ipairs(inlines) do t[#t + 1] = x end
  for i = 1, #t do
    if t[i].t == "Str" and t[i].text == "DAgger-style" then
      local j = i + 1
      while j <= #t and t[j].t == "Space" do j = j + 1 end
      if j <= #t and t[j].t == "Str" and t[j].text == "relabel" then
        table.insert(t, j + 1, pandoc.RawInline("latex", "~\\citep{ross2011dagger}"))
        dagger_done = true
        return t
      end
    end
  end
  return nil
end

-- 2 & 3) Title + abstract extraction happen at the whole-document level so we
--        can look across the block stream.
function Pandoc(doc)
  local body = {}
  local abstract = {}
  local grabbing_abstract = false
  local title_set = doc.meta.title ~= nil

  for _, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.level == 1 then
      -- First H1 becomes the title (unless one was supplied via metadata);
      -- either way it is removed from the body.
      if not title_set then
        doc.meta.title = pandoc.MetaInlines(b.content)
        title_set = true
      end
    elseif b.t == "Header" and b.level == 2 and stringify(b) == "Abstract" then
      grabbing_abstract = true
    elseif grabbing_abstract and b.t == "Header" and b.level <= 2 then
      -- Reached the next section: stop capturing and keep this header.
      grabbing_abstract = false
      table.insert(body, b)
    elseif grabbing_abstract then
      table.insert(abstract, b)
    else
      table.insert(body, b)
    end
  end

  if #abstract > 0 then
    doc.meta.abstract = pandoc.MetaBlocks(abstract)
  end

  return pandoc.Pandoc(body, doc.meta)
end
