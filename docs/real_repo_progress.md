# Real-repo experiment: overnight progress (2026-06-30 → 07-01)

> **Historical lab log.** This file is a chronological record of the real-repo follow-up work, including
> hypotheses and "next steps" that were later resolved. It should be read as provenance for `REPORT.md`,
> not as current project status. In particular, its later “types, not navigation” and checker-null language
> is superseded by the evidence audit: the controlled whole-file-read result remains supported, while the
> dispatch and authoring results are tentative because of leakage, opportunity, and integration confounds.

Morning summary of what got built and what it found. Design + validity invariant are in
`docs/real_repo_plan.md`. No matrix runs yet (that needs the env-setup step below); this pass built and
validated the pipeline and produced the auditable candidate pool.

## What's built and working

1. **Harness** — recovered `scaffold/real_env.py` (`RealRepoEnv` + AST `SymbolResolver`) from git
   history. It already exposes `read_file`, `goto_definition`, `find_references`,
   `pyrefly_diagnostics` (= check_types), and `run_tests` over a checked-out repo, with a live
   `pyrefly lsp` daemon and AST fallback. Imports cleanly and passes its self-check in the current tree.
2. **Loader** — `scripts/realbench/swe_loader.py`: loads SWE-bench Verified, keeps a per-repo blobless
   clone cache, checks a task out at `base_commit`, applies the task's `test_patch`, builds the oracle
   pytest command over FAIL_TO_PASS/PASS_TO_PASS. Validated end-to-end on a requests task in ~9s.
3. **Selection scanner** — `scripts/realbench/select.py`: implements the S1–S3 validity invariant
   (cross-file dependency / non-trivial symbol / expensive counterfactual) purely from the gold patch +
   an AST resolve over the checked-out source (no dependency install). Test files are excluded from
   resolution so a symbol never resolves to a test.
4. **Ranker** — `scripts/realbench/rank.py`: merges the per-repo scans into one ranked candidate pool
   (`runs/realbench/candidates.json`).

## What the scan found

Scanning ~22 tasks per repo across the SWE-bench Verified repos, **42 of 183 scanned tasks are
admissible** (pass S1&S2&S3), across 7 repos (sympy 10, xarray 10, matplotlib 6, astropy 5, sphinx 5,
django 4, pytest 2) — plenty to hand-audit ~15 from. The admissible tasks have exactly the cost-gap
structure the experiment needs: the fix, in file A, depends on a symbol defined in a **large** other
file B, so a read-only agent pays to read B whole while `<defn>` fetches just the span. Top examples:

| task | cross-file symbol | defined in | edits |
|---|---|---|---|
| astropy-13398 | `CartesianRepresentation` (type) | representation.py (3474L) | builtin_frames/__init__.py |
| sympy-13878 | `Zero` (type) | core/numbers.py (3843L) | stats/crv_types.py |
| matplotlib-20859 | `Axes` (type) | axes/_axes.py (8151L) | legend.py |
| xarray-7229 | `Dataset` (import) | core/dataset.py (9214L) | core/computation.py |
| django-11149 | `ManyToManyField` (type) | fields/related.py (1647L) | admin/options.py |
| sphinx-11510 | `Sphinx` (type) | application.py (1362L) | directives/other.py |

The full ranked pool with per-criterion evidence (which symbols, which files, sizes, F2P count) is in
`runs/realbench/candidates.json` (per-repo scans in `runs/realbench/scan/`). `requests` and the early
`pytest` tasks correctly scored zero: those fixes are single-file with only stdlib/local symbols, so
there is genuinely no cost gap to measure — the scanner is discriminating, not just permissive.

## Update: env set up + full pipeline built and validated (2026-07-01)

**Environment is unblocked.** The host is ARM64 (DGX Spark GB10); SWE-bench's prebuilt images are
x86_64, so they fail natively. Registering qemu emulation once
(`docker run --privileged --rm tonistiigi/binfmt --install amd64`) plus
`DOCKER_DEFAULT_PLATFORM=linux/amd64` makes them run. **The gold patch for `django__django-11149`
resolves through the full oracle** (`scripts/realbench/score.py`, ~90s emulated), so test execution and
scoring work end to end.

**Pipeline complete.** Added `scripts/realbench/run_matrix.py` (the agent runs on a host checkout under
the three tool conditions R/D/I; the final `git diff` is the prediction) and `scripts/realbench/score.py`
(scores predictions via the swebench Docker oracle). Every stage is now built and validated.

**Refined candidate pool.** Added an S6 tractability filter (small edit-site file + small gold patch) so
the pool favours localizable tasks. Of 42 admissible, **14 are audit-ready**: a small file A edited using
a symbol defined in a large file B. Best examples (ideal cost-gap): `astropy-14182` edits a 66-line file
using `QTable` (4247L); `django-11206` edits a 79-line file using a symbol in a 907-line file;
`xarray-4356`, `sphinx-8265`. These are in `runs/realbench/candidates.json` under `audit_ready`.

**Honest design finding from a smoke run.** A score-only rollout (no test feedback) on the *hard*
`django-11149` (1600-line admin file) flailed: the agent read large files, blew context to ~380k tokens,
and produced no edit. Two things this tells us, and a fork for you:

- **Task tractability matters** — the S6 filter above is the response; the audit-ready pool avoids
  huge edit-site files.
- **Test feedback is probably needed** — real fixes want iteration. Options: native host venvs for the
  pure-Python repos (django/sympy/sphinx/pytest run natively on ARM64, fast) so `run_tests` is cheap in
  the loop; emulated Docker `run_tests` is correct but ~90s/run, too slow to iterate.
- **The design fork (your call):** (a) **retrieval-isolated** — give the agent the fix location (I wired
  a localization hint from the gold hunks) and measure only the cost of retrieving the cross-file symbol
  (clean, directly tests our claim, mildly semi-constructed); or (b) **full-agentic** — the agent
  localizes + retrieves + fixes (realistic but noisy, and needs test feedback to succeed). I lean (a) for
  a clean efficiency measurement; (b) is a bigger, separate result.

I did **not** run the matrix for real (it needs the fork decided + the audited task set); ~$0.29 of API
spend on smokes, no more.

## Pilot result (2026-07-01): score-only does not work; test feedback is required

Ran a real pilot on the auto-selected audit-ready tasks (invariant-selected, not human-audited, as
agreed): astropy-14182, django-11138, sphinx tasks, with `claude-sonnet-4.5` under R/D/I and tight read
caps. The definitive finding:

- **A score-only agent (no test feedback) does not resolve real tasks.** On django-11138 the read-only
  patch was **unresolved** (wrong fix) and the go-to-definition run produced an **empty patch** (no
  edit); both hit the turn limit without converging. The gold patch resolves through the same oracle, so
  this is the agent, not the harness. Frontier models cannot one-shot these fixes from the issue + code
  alone; they need to iterate against the failing test.
- **Rollouts are expensive.** Each real-repo tool-calling rollout is ~200k input tokens (the API re-sends
  the growing context every turn, times multi-turn exploration), ~$0.6 at sonnet pricing. A full matrix
  (8 tasks x 3 conds x 2 models x 2 seeds) would be $20-40 for data that, score-only, does not resolve.
- **Emulated scoring is fast for light repos, slow for heavy ones.** django/sphinx score in ~90s; astropy
  (heavy test suite under qemu) takes 5+ minutes. Favour pure-Python fast-test repos.
- **Early efficiency signal is confounded**, not clean: on astropy-14182 the D (defn) run used *more*
  tokens than R because it hit the turn limit, and the agent reads broadly for context regardless of
  defn, diluting the retrieval saving. We cannot conclude the efficiency win from score-only data.

**Conclusion and the path that will work:** the experiment needs **test feedback in the loop** so the
agent converges, which on this ARM64 host means **native venvs** for the pure-Python repos (django,
sphinx, sympy, pytest install and test natively and fast on ARM64; no emulation for the loop, Docker only
for the final canonical score). That is the next build: per-repo venv setup (reuse the swebench install
spec), route `RealRepoEnv.run_tests` to it, give the agent `run_tests`, then re-run the matrix. Budget it
at ~$0.5-1.0/rollout, so a focused 8-task x R/D/I x 2-model x 2-seed run is ~$30; keep it tight. Total
spend so far: ~$2.60 (smokes + this pilot), no more until this is decided.

## mini-swe-agent + LSP-tool ablation: built, and the smoke undercuts the premise (2026-07-01)

Per the decision to use a real, credible agent, wired up **mini-swe-agent** (Princeton SWE-agent
team's minimal bash-only agent; strong SWE-bench Verified scores) and ablated the language server
as a **shell command**, which fits its bash-only design exactly:

- `scripts/realbench/codenav.py` - standalone, pure-stdlib CLI: `codenav defn SYMBOL` (go-to-def
  span) and `codenav refs SYMBOL` (find-references). Runs inside the task container against
  `/testbed`; grep-prefilter keeps it cheap on big repos; indentation-based span extraction is
  py3.6+ safe (no `ast.end_lineno`).
- `scripts/realbench/mini_ablate.py` - two-arm driver. **off** = stock mini-swe-agent; **on** =
  identical + `codenav` base64-injected via the `env_startup_command` hook + one prompt paragraph
  advertising it as the cheap way to retrieve a definition. Records per arm: input tokens (sum of
  `prompt_tokens`), peak context, model calls, whether `codenav` was actually called, patch.
  Runs inside the SWE-bench container (test feedback free; no native-venv build needed).

**Integration is validated end to end** (zero-API container check + a $0.65 one-task run):
codenav injects and resolves real django symbols under the container's py3.6/ASCII locale
(this caught a UnicodeEncodeError, now fixed); OpenRouter routing works; token usage (incl. real
`cost`) is captured; patches are emitted in swebench format ready for `score.py`.

**But the one-task smoke (django-11138, sonnet, both arms) undercuts the efficiency premise.**
The `on` arm **never called `codenav`** despite it being installed and advertised. Sonnet's 40
actions were 17x `sed -n 'X,Yp'` (ranged reads) + 15x `grep` + 6x `cat`. A capable agent in a
bash scaffold **already self-retrieves cheaply**: it greps to localize, then reads a narrow line
range - it does go-to-definition by hand with primitives it trusts. So the counterfactual the
efficiency claim rests on - *the agent pays for an expensive whole-file read* - **does not occur
for a strong bash agent**. The synthetic 3.5-4.7x was measured against a baseline whose only read
action was whole-file; a real agent never takes that action, so the language server's token
advantage over it largely evaporates. Token ratio here was 0.90x (the `on` arm used slightly
*more*, doing more of the same), not a saving.

Caveats (why this is a signal, not yet a headline): (1) one task, one model, one seed; (2) the
40-step cap was too tight - both arms hit `LimitsExceeded` with empty patches (mini-swe's default
is 250), so there is no matched-success cell here; (3) django-11138 had no genuine large-def need
anyway - the scanner's flagged cross-file dep (`lower` in defaultfilters.py) was a **false
positive** (the real fix is a timezone bug in `db/backends/*/operations.py`). To confirm the
finding, the right next run is 2-3 tasks with a *genuine* large-type-definition need (astropy-14182
`QTable` 4247L, sphinx-8265 `Index`, sympy-14531 `Normal`) at a higher step limit, to see whether
the agent ever does a whole-file read that `codenav` would replace, or whether grep+ranged-read
always dominates. Prediction: small-to-zero gap. Spend so far this build: ~$0.65.

## Prompt-framing confirming run: election is elicitable, but the token payoff is not (2026-07-01)

Followed up the smoke's premise problem with the prompting question it raised: the report's own
finding is that a capable model elects the cheap action *when the system prompt frames it that
way*. So a third arm was added and run on three genuine large-definition tasks (astropy-14182
`QTable` 4247L, sympy-14531 `Normal` 2987L, sphinx-8265 `Index`), one seed each:

- **R (off)** bash only; **D (on)** codenav available, mild instance advertisement; **D+ (onx)**
  codenav + a STRONG system-prompt directive to prefer `codenav defn/refs` over reading files.

Per-arm (`codenav` = defn+refs calls, `cat_whole` = whole-file .py reads, the codenav-replaceable
action; in/call = avg input tokens per model call, which normalises for step count):

| task | arm | in/call | codenav | cat_whole | exit | patch |
|---|---|---:|---:|---:|---|---:|
| astropy | off | 16952 | 0 | 3 | Submitted | 1508B |
| astropy | on | 16721 | 1 | 3 | Limit | 0 |
| astropy | onx | 17975 | 7 | 2 | Limit | 0 |
| sympy | off | 19763 | 0 | 2 | Submitted | 3246B |
| sympy | on | 19742 | 0 | 2 | Limit | 0 |
| sympy | onx | 18651 | 5 | 1 | Limit | 0 |
| sphinx | off | 22951 | 0 | 1 | Limit | 0 |
| sphinx | on | 17217 | 8 | 0 | Limit | 0 |
| sphinx | onx | 17262 | 0 | 3 | Limit | 0 |

**What holds up (robust to the confounds below):**

1. **Election is prompt-liftable, even in a real bash agent.** Strong system framing raised codenav
   use where mild advertisement did not: astropy 0/1/**7**, sympy 0/0/**5**. Consistent with the
   report's election finding, now shown in mini-swe-agent. (Noisy at one seed: sphinx went 0/**8**/0
   - mild used it heavily, strong not at all.)
2. **Eliciting codenav use does NOT lower token cost.** Normalised per-call input tokens do not drop
   when codenav is used: astropy-onx (7 codenav) is *fatter* than off; sympy-onx (5) is ~5% leaner;
   sphinx's two lean arms include onx which used codenav **0** times. No consistent codenav->savings
   effect. The raw off/arm ratios (0.69-1.33x) are confounded by termination (only R arms converged)
   and are not a clean measurement.
3. **The whole-file-read counterfactual barely occurs.** `cat_whole` is 0-3 across all 9 arms (of
   44-60 actions each); retrieval is dominated by `grep` + ranged `sed -n`. The expensive read that
   the synthetic 3.5-4.7x beat is simply not what a capable bash agent does.
4. **Self-retrieval is sufficient to solve.** astropy-14182's R arm **submitted a real patch**
   (1508B) in 44 calls using 8 grep + 2 sed + 3 cat + **0 codenav** - it fixed a task involving
   `QTable` (4247L) without ever reading QTable's definition or using the language server.

**Bottom line for the report framing.** The prompting lever works (election is capability/prompt
gated - confirmed in the wild), but pulling it buys no efficiency in a real bash agent, because the
counterfactual is grep+ranged-read, not a whole-file read. The retrieval-efficiency headline is
real only against a forced-whole-file-read baseline (our synthetic env); it does not transfer to a
capable agent with shell primitives. Recommend scoping the efficiency claim accordingly.

Caveats: one seed/cell; 3 tasks; one model (sonnet); step cap 60 so only 2/9 arms converged (no
matched-success cell, hence per-call tokens / behaviour are the reliable lenses, not raw ratios).
Spend: ~$5.65 (confirm) + $0.65 (smoke) = ~$6.30.

**Mechanism (why eliciting codenav buys nothing): it is additive, not substitutive.** Cross-checking
each `codenav defn SYMBOL` against the trajectory's read commands, in ~16 of 18 defn calls the agent
**also read the same file it just defn'd**, via grep/sed/cat - often heavily:
- sympy-onx defn'd `StrPrinter` / `_print_Limit` / `_print_Relational` (all in `str.py`) and still
  `sed -n`'d `str.py` **22 times**.
- astropy-onx defn'd `RST` / `SimpleRSTHeader` / `SimpleRSTData` (all in `rst.py`) and still `cat`'d
  the **whole** `rst.py`.
- sphinx-on defn'd `unparse` -> `ast.py`, then `sed -n '71,250p' ast.py` (a 180-line read); touched
  `ast.py` 13 times.

A definition *span* is not what makes a fix: the agent needs the edit site, surrounding context,
tests, and call sites - i.e. a read of the file - which the span does not provide. So `codenav defn`
does not displace a read; it is an extra call on top of the reads the agent does anyway. This is the
mechanism behind the null token result, and why the heaviest-codenav arm (astropy-onx) was *fatter*,
not leaner. The efficiency premise ("defn replaces an expensive whole-file read") fails twice over:
the agent rarely whole-file-reads, and when it defn's it reads the file regardless.

## Dispatch-ambiguity experiment: x86 host (chunklebox) + pyrefly type-aware goto, validated (2026-07-02)

qemu on the ARM host cannot run pyrefly (the x86 binary segfaults even on `--version`), so the
in-container pyrefly path is impossible under emulation. Moved to **chunklebox** (native x86, 16-core,
Ubuntu 24.04, Docker 29). There the whole stack runs natively, no qemu: SWE-bench images run for real,
mini-swe-agent runs in-container with real test feedback, AND pyrefly runs in-container. Ported the repo
(rsync), lean venv (`datasets`, `swebench`, `mini-swe-agent`, `openai`, `pyrefly` 1.1.1 x86).

**`scripts/realbench/pyrefly_nav.py`** is the treatment tool: an in-container CLI wrapping the (validated)
pyrefly LSP client, `pyrefly_nav goto FILE LINE SYMBOL` (type-aware, receiver-correct go-to-definition)
and `impls` (find-implementations). Validated in a native django container:
- **Receiver-aware goto works.** Synthetic `x.foo()` with `x: Sub` -> resolves to `Sub.foo`, not
  `Base.foo` or the unrelated `Other.foo`. Real cross-file: `self.to_python(value)` ->
  `django/db/models/fields/__init__.py:560` (the class's override), ~6s incl. cold index. This is the
  exact thing grep cannot do (grep `def to_python` returns ~24 candidates).
- pyrefly advertises `implementationProvider: True`, but `impls` is not the differentiator: **grep
  `def NAME` already enumerates the override set**, so for dynamic-dispatch cases (sympy printers via
  `getattr`, django `connection.ops` chosen at runtime) grep ~ find-implementations and the LSP adds
  nothing. The LSP's UNIQUE win is single-target goto on a **static receiver**.

**Experiment design (scoped by the above).** Arms: **G** = mini-swe-agent bash only (grep/sed); **T** =
G + `pyrefly_nav goto`. Task set: static-receiver dispatch where the fix edits one override among many,
from `dispatch_candidates.json` — django-11211 (`get_prep_value`, 21 Field classes), sklearn-10908
(`inverse_transform`, 24 transformers), sympy-12419 (`_eval_derivative`, 28), plus django-11138. Primary
metric: **mis-localization** (does the agent read/edit the RIGHT override vs the gold hunks) — needs no
test run; secondary: resolved@1 via the in-container tests / swebench score. Include a weaker model
(effect predicted capability-gated). Honest prior: a strong agent may disambiguate by reading a few grep
hits, so T may beat G on mis-localization/steps more than on final correctness.

### First G-vs-T smoke (django-11211, sonnet, native x86): precision is redundant too

Ran `mini_ablate.py --arms off,nav` on django-11211 (`get_prep_value`, edit one of 21 Field overrides)
natively on chunklebox. Native x86 fixes the convergence problem: both arms **Submitted** and both
**RESOLVED 1/1** (held-out tests pass) - unlike the emulated ARM runs that hit the step limit empty.

- **off (grep only): RESOLVED.** Sonnet edits the correct override via 15 grep + 8 sed + 4 cat, no
  type-aware tool.
- **nav (pyrefly_goto available + advertised): RESOLVED, but `pyrefly_goto` called 0 times.** Sonnet
  disambiguated by reading; it never reached for the tool. T == G behaviourally; the 0.82x token
  difference is run variance.

So precision-under-ambiguity joins information and efficiency: **redundant for a capable self-reading
agent.** It resolves a dispatch-ambiguous task by reading and does not use (or need) the receiver-aware
tool even when handed it. Note: since `off` already resolves 1/1, strong framing (navx) cannot raise
resolved@1 here - the only place LSP precision can change an OUTCOME is a task the model gets WRONG via
mis-localization, which points at a WEAKER model. Spend: off $0.76 + nav $0.89 = ~$1.65; both scored
native (no emulation). Driver arms `nav`/`navx` + score.py dict-preds fix committed.

**Reframing (the piece this does NOT settle): efficiency.** The django-11211 result says the agent solves
the *information/correctness* problem by reading. It says nothing about *efficiency*, because sonnet
never USED pyrefly_goto, so there is no token comparison. And the dispatch domain is exactly where goto
might beat grep+sed: `grep def get_prep_value` returns 21 candidates the agent must wade through, whereas
goto returns the one. Whether that saves tokens is the open efficiency question, and (per the report's
own §3-4) it is capability- and policy-gated: a strong model won't elect the cheap action without being
pushed. So the natural next probe uses the LOCAL model (free, and trainable), mirroring the report's
efficiency/election/DAgger arc in this real domain: base solve with grep+sed, then LSP available, then
prompt to prefer it, and worst case a DAgger relabel to push toward it, measuring tokens at matched
success. Architecture (reuses the working chunklebox setup): serve the local Qwen on the GB10 (vLLM),
run mini_ablate on chunklebox against that endpoint over Tailscale (100.78.95.73), so containers +
pyrefly + tests stay native x86 and only inference is remote. Recon: vLLM not yet installed on the GB10;
Qwen2.5-Coder-{7B,14B} and smaller are cached; a prompt-electing model (27B-class) tests efficiency
without training, a 7B needs the DAgger step. Not started; this is the next build.

### Dispatch-efficiency experiment: harness + 7B smoke (2026-07-02)

Built the local-model dispatch-efficiency probe (no vLLM needed: the transformers harness already runs
Qwen locally and does DAgger). Two harness additions to `scaffold/stream_agent.py`, both committed:
- **Realistic baseline actions.** `<grep pat="REGEX"/>` (textual search over the repo, so a method on N
  classes returns N hits to disambiguate) and `<read path lines="a-b"/>` (ranged `sed -n`). Previously the
  only read was a whole-file `<read>`, i.e. the strawman baseline; now the baseline is what a shell agent
  actually does. Plus a `sys_override` hook so the runner dictates the per-condition tool advertisement.
- **`<defn>` anchor fix.** In line-edit mode `<defn>` now returns the resolved class span WITH its file
  path and real line numbers (`_resolve_defn` returns `(src, path)`, `_fmt_defn` numbers it). Search mode
  is byte-identical, so the report's existing experiments are untouched.

Tasks (`scripts/realbench/dispatch_tasks.py`): 3 self-contained on-disk repos where a method is overridden
on ~10 classes, a typed receiver in `app.py` calls it, exactly one override is buggy, and a pytest fails
at base / passes on the right fix. `codec_serialize` (serialize on 10 handlers, receiver JsonHandler),
`field_validate` (validate on 10 fields, receiver EmailField), `node_to_str`. Runner
`scripts/realbench/local_dispatch.py`; conditions `grep_base` (grep+ranged-read, no defn), `defn_avail`
(+LSP `<defn>`, neutral), `defn_prompt` (defn framed as the cheap way to pick the right override). Metric:
`in_toks` at matched success + election counts.

7B smoke (Qwen2.5-Coder-7B, run BEFORE the anchor fix, so the defn cells are invalid but still informative):

```
grep_base   resolved=1/3  mean_in_toks=1827 (resolved=1371)  n_defn=0.0  n_grep=1.3
defn_avail  resolved=0/3  mean_in_toks=4880                   n_defn=1.0
defn_prompt resolved=0/3  mean_in_toks=4204                   n_defn=3.3
```

Findings: (1) the 7B is weak on the line-edit protocol (only field_validate solved via grep, cleanly at
1371 tokens with 2 grep + 2 ranged reads; codec flailed to the read budget, node looped without editing).
(2) **Prompt framing lifts defn election** even in the 7B: mean n_defn 1.0 -> 3.3 neutral -> prompted. (3)
Every defn cell FAILED, but purely from the anchor bug (the model elected defn, got the right class, then
could not anchor a line edit and thrashed) - fixed and re-running. Reframe the metric this suggests: for a
weak model the LSP's value here looks like RESOLUTION (handing it the right class among 10, which grep+read
forces it to REASON to) more than token efficiency, since a clean grep solve (~1371) and a clean defn solve
are comparable in tokens. Next: 27B (Qwen3.6-27B) with the fix across all three conditions (running), then
7B + DAgger. Caveat to watch: the override classes are small (~10 lines), so a precise-reading model's
grep baseline is already cheap and defn saves little; if the effect needs amplifying, enlarge the classes.

### Dispatch-efficiency results: 27B redundant, 7B edit-bound (2026-07-03)

**27B (Qwen3.6-27B, with the anchor fix): efficiency REDUNDANT in the dispatch domain.**

```
grep_base   resolved=3/3  mean_in_toks=1385  n_grep=0.0  read_whole=1.0
defn_avail  resolved=3/3  mean_in_toks=1386  n_defn=1.0
defn_prompt resolved=3/3  mean_in_toks=1335  n_defn=1.0
matched-success ratio grep_base/defn_prompt = 1.037
```

The competent model solves every task under every condition, and the mechanism is the point: in
grep_base it did ZERO greps. It read `app.py`, saw the receiver's type annotation (`x: EmailField`), and
went straight to the one relevant class. The dispatch ambiguity that makes `grep def NAME` return 10
candidates never bit it, because the type is readable at the call site. It elects `<defn>` when offered
(and framing shifts it further off reads), but that changes neither tokens (1.04x) nor resolution (3/3).
This is the dispatch-domain confirmation, for EFFICIENCY, of the report's thesis: a capable model reads
the receiver type and self-localizes, so the language server's single-target resolution is redundant. The
1.04x sits against the synthetic 3.5-4.7x precisely because the realistic baseline (read the ONE relevant
class after reading the type) is already cheap, unlike the whole-file-read strawman.

**7B (Qwen2.5-Coder-7B, with the anchor fix): retrieval is not the bottleneck, edit competence is.**

```
grep_base   resolved=1/3 (field_validate 1371)
defn_avail  resolved=1/3 (node_to_str 1281 clean)
defn_prompt resolved=0/3
```

temp=0, so this is deterministic behaviour, not sampling noise. Two real effects: (1) `<defn>` RESCUED
localization on node_to_str, where grep_base looped without ever editing (0 edits) but the anchored defn
gave the 7B the exact class + line numbers and it solved cleanly in 1281 tokens / 33s; (2) `<defn>` HURT
field_validate, which grep_base solved (1371) but defn_avail thrashed (4535), and strong framing
(defn_prompt) pushed it onto defn everywhere and it thrashed to 0/3. So the 7B already ELECTS defn when
framed (election was never its problem, unlike the report's synthetic single-file tasks where DAgger fixed
election); its ceiling is multi-file line-edit competence, which the retrieval method does not move. Net
resolution ~1/3 under every condition. The resolution-assist (defn rescuing localization) is real but 1
help / 1 hurt / 1 both-fail on 3 tasks is too few. Decision (Ian): EXPAND to 15 dispatch tasks (varied
method family, receiver type, override count, localization difficulty) and re-run 7B + 27B, to firm up or
kill the weak-model resolution-assist claim. Task expansion in progress; all local compute, no API spend.

### 15-task run: capable model efficiency-neutral, weak model below threshold (2026-07-03)

Expanded to 15 dispatch tasks (override counts 8-15, small-to-large bodies, tempting-similar siblings,
varied bug types; all pass GATE 1, pyrefly resolves the receiver-correct override for each). Ran the 7B
and 27B across all 15 x 3 conditions (temp=0). Runner now reports a paired per-task efficiency delta.

**27B: LSP efficiency-neutral in the dispatch domain, robustly, at N=15.**

```
grep_base   resolved=15/15  mean_in_toks=1436  n_grep=0.3  read_whole=1.0
defn_avail  resolved=14/15  mean_in_toks=1552  n_defn=0.8
defn_prompt resolved=15/15  mean_in_toks=1380  n_defn=1.0
paired delta defn_avail  vs grep_base: n=14  ratio=0.972  mean_delta=-43 toks (defn slightly costlier)
paired delta defn_prompt vs grep_base: n=15  ratio=1.041  mean_delta=+56 toks (defn slightly cheaper)
```

The token ratio is ~1.0 either way (0.97 to 1.04); per-task deltas are small and mixed (+/-50 to 350),
with occasional thrash outliers in BOTH directions (defn_avail thrashed resource_cost to 3390 and lost
that one task; defn_prompt thrashed job_priority to 2387). Mean is a wash. Mechanism confirmed at scale:
grep_base n_grep=0.3 - the 27B barely greps even when grep is the advertised tool; it reads the receiver
type annotation and goes straight to the one class (read_whole=1.0). So go-to-definition, whether elected
neutrally or under framing, neither helps nor hurts token cost on average, and occasionally causes a
thrash the plain read avoids. The synthetic 3.5-4.7x efficiency win does NOT transfer to the dispatch
domain for a capable model, because the realistic baseline (read the type, read the one class) is already
cheap.

**7B: below the competence threshold, no reliable LSP benefit, forcing defn hurts.**

```
grep_base   resolved=3/15  mean_in_toks=2508  n_defn=0.0
defn_avail  resolved=4/15  mean_in_toks=2848  n_defn=2.7
defn_prompt resolved=2/15  mean_in_toks=3545  n_defn=1.0
paired delta: NO task resolved in BOTH grep_base and any defn condition -> efficiency delta UNDEFINED
```

The solved sets are DISJOINT (grep_base: field_validate, record_to_dict, rule_matches; defn_avail:
encoder_encode, node_to_str, order_compute_total, row_format_row; defn_prompt: animal_describe,
row_format_row). defn_avail's 4 vs grep_base's 3 is not the LSP rescuing grep's failures - the 7B solves
a different random subset. The node_to_str rescue from the 3-task run was one of the coin-flips and did
NOT generalize: at N=15 success is sparse (2-4/15) and uncorrelated with the retrieval method, because the
7B's bottleneck is multi-file edit competence, not retrieval. defn_prompt (strong framing) is WORST on
both resolution (2/15) and cost (3545 tokens): pushing a below-threshold model onto defn just adds thrash.

**Synthesis across three capability tiers (dispatch domain).** Frontier (sonnet, section 3.5): resolves
by reading, does not elect the tool. Capable local (27B): resolves by reading (self-localizes on the
receiver type), elects defn when offered but efficiency-neutral (ratio ~1.0) and occasionally thrashes.
Weak local (7B): below threshold, no reliable resolution benefit, forcing defn hurts. The efficiency
question is answered with a clean negative: in the domain built to favour the language server, its
go-to-definition buys no token efficiency for a capable model, because reading the receiver type is the
cheap path the model already takes. Note on DAgger: election was never the 7B's gap here (it elects defn
when framed), so the report's election-DAgger lever does not apply; the gap is edit competence.

### The reframe: types are load-bearing, LSP navigation is not (2026-07-03)

Ian's reframe of the negative. The 27B localizes for free because the code carries a correct type
annotation, which it reads (the grep_base trajectories confirm the FIRST action opens the one buggy
override file, having read the receiver type in app.py). So the fact go-to-definition would compute,
which override binds, is already written in the source as a type the model reads. The argument closes:
if dispatch is statically resolvable the type is in the source and the agent reads it, so navigation is
redundant; if it is not statically resolvable it is dynamic, and a static goto cannot resolve it either.
Navigation never beats a readable type. So the value is not in the LSP's navigation, it is in the TYPES
themselves, and a type checker's contribution is keeping those annotations correct so the code stays
self-describing, not feeding the agent a live lookup. This has been folded through REPORT.md (status,
abstract, 3.5, 7, conclusion). New research line: (1) how much is a readable annotation worth (strip it);
(2) does a type checker help when AUTHORING new code (a different regime from the section-5 bug-fix test).
Memory: [[project_types_pivot]].

### Exp 1: receiver-type ablation (annotated / stripped / indirection)

Added a `--typing` ladder to the dispatch suite (dispatch_tasks.py build_tasks(typing=), local_dispatch.py
--typing). Same 15 tasks, same code, varying the DISTANCE from the call site to the type:
- L0 annotated (current): `def run(x: Buggy, ...)`. Type at the call site.
- L1 stripped: `def run(x, ...)`. Type only in the test's construction `run(Buggy(), ...)`, one hop.
- L2 indirection: `x = make_recv()` with `def make_recv() -> Buggy`. Type behind a return-annotated
  factory, a trace.

**pyrefly-resolution validation (build-time, before the model run) is itself the crux result:**

```
variant         base/gold/grep      pyrefly goto resolves the right override
L0 annotated    pass (15x3)         15/15  True
L1 stripped     pass                 0/15  (returns NOTHING: a bare param's type is not inferable)
L2 indirection  pass                15/15  True
```

So the LSP can substitute for a missing call-site annotation ONLY when the type stays statically
inferable (L2, via the factory return), NOT when it is knowable solely from the test's runtime
construction (L1). This is the mechanism behind the reframe, shown at the tool level. Now running the
27B across L1 + L2 x {grep_base, defn_avail, defn_prompt} (90 rollouts) to measure the AGENT side:
(a) does grep_base token cost rise L0 -> L1 -> L2 (how much the annotation is worth to the agent); (b) at
L2, where goto resolves, does defn beat grep_base (the one spot the LSP could earn its keep by skipping
the factory trace); (c) at L1, where goto returns nothing, does the useless defn cause thrash or a clean
fallback. All local compute, no API spend.

**27B L1 stripped result:**

```
grep_base   resolved=14/15  mean_in_toks(resolved)=1429  n_grep=0.3   (L0 annotated was 1436, 15/15)
defn_avail  resolved=15/15  mean_in_toks=1452  ratio grep/defn=0.984  (goto returns NOTHING at L1)
defn_prompt resolved=14/15  mean_in_toks=1512  ratio grep/defn=0.945
```

Findings: (a) stripping the call-site annotation does NOT degrade the 27B (1429 vs L0's 1436, 14/15 vs
15/15) - it reads the receiver type from the test's construction `run(Buggy(), ...)` instead of the
annotation. So it is READABLE TYPE INFO (annotation OR construction site), not the annotation specifically,
that is load-bearing. (b) defn is genuinely useless at L1 (pyrefly returns nothing), yet the capable model
handles it gracefully: defn_avail still resolves 15/15 by falling back to reading (no thrash), at ~1.0x
cost; only when PROMPTED to prefer the dead tool (defn_prompt) does it pay a small penalty (0.945, i.e.
grep slightly cheaper). A capable model tolerates a useless language server.

Design caveat (important, applies to L2 too): `build_prompt` shows the FULL app.py (the call site) plus
the failing test in the prompt, so the receiver type is never actually HIDDEN from the model - it is
always somewhere in the shown source (annotation at L0, construction at L1, factory at L2). So this ladder
tests type-LOCATION-within-readable-source, not type-that-requires-retrieval. That is why L1 does not
degrade: the type just moved from app.py to the test, both shown. A stronger test of "how much does
retrieval of the type cost" would HIDE app.py and force the agent to fetch it; noted as a possible follow
up. Even so the result is consistent with the reframe: wherever the type sits in readable source, the
capable model reads it and the LSP is redundant (and at L1 the LSP cannot even resolve).

**27B L2 indirection result (goto CAN resolve here, 15/15):**

```
grep_base   resolved=14/15  mean_in_toks(resolved)=1465  n_grep=0.3
defn_avail  resolved=15/15  mean_in_toks=1514  ratio grep/defn=0.982
defn_prompt resolved=15/15  mean_in_toks=1403  ratio grep/defn=1.065
```

The crux test lands negative. Even at L2, where the type is behind a factory (a genuine trace for grep)
AND pyrefly goto resolves it 15/15, go-to-definition does NOT beat grep_base: the ratio is ~1.0 (0.98
available, 1.07 prompted), with defn occasionally causing a big thrash (job_priority: grep 1526 vs
defn_avail 2673). The 27B reads the factory in the shown app.py and traces the type just as cheaply.

**Full ladder, 27B grep_base mean_in_toks(resolved): L0 1436 -> L1 1429 -> L2 1465.** Essentially FLAT.
Moving the receiver type farther from the call site (annotation -> construction -> factory indirection)
does not raise the capable model's cost, and defn stays ~neutral (0.98-1.07) at every rung, including the
one where it resolves. Conclusion of Exp 1 as designed: the type's LOCATION does not matter when the
source containing it is shown, because the model reads it. This confirms the reframe robustly, with the
stated caveat that the harness always shows app.py + the test, so the type is never truly HIDDEN. To
measure the cost of RETRIEVING the type (the only regime where goto could save a read or two), a variant
that hides app.py and forces the fetch is needed. But note the LSP's ceiling is intrinsically small: for a
capable model any statically resolvable type is readable in one or two reads, so goto can save at most
that. This bounded ceiling is the deep reason the reframe holds and why the more interesting next question
is the type CHECKER at authoring time (Exp 2), not squeezing the navigation result further.

### Exp 2: does a type checker help when AUTHORING new code? (design, 2026-07-04, in progress)

The reframe said a type checker's value is keeping the code's types correct. Section 5 tested the checker
as in-loop info on BUG FIXES (gapd2: fill one small stub, a designed plausible-wrong alternative the
checker would catch) and found it redundant: frontier models write the correct fix anyway. Authoring is a
different regime with a bigger, ORGANIC type-error surface (undefined names, wrong signatures, bad
imports, arity, attribute access across several interacting components), which is where fast checker
feedback might finally pay. Exp 2 tests it.

Design: an `authoring` suite (~12 tasks). Each task is a typed STUB (function/class signatures + docstring
spec, bodies raise NotImplementedError) plus an optional provided typed `lib.py` the target must use
correctly, a VISIBLE example test the agent may run, and a HELD-OUT scoring test (as in gapd2, so
behavioural correctness is not trivially confirmed by the visible test). The tasks are larger than gapd2
(several functions/classes, imports, interacting types) so type errors occur organically. Arms toggle only
the type checker: `none` (implement + run visible test), `check` (a check_types() tool the agent may
elect), `feedback` (pyrefly diagnostics volunteered automatically after each edit). Metrics: held-out
pass@1 (primary), visible pass, edits/iterations to green, residual type-error count (pyrefly on the final
submission), input tokens. Models: local 27B and 7B (free, capability contrast; the hypothesis is the
checker helps a weaker model that errs more), with a frontier run as an optional section-5 comparison.
Harness reuses section 5 (api_agent check_types / synth_mf + stream_agent gapd path).

Built: `scripts/synth_tasks_authoring.py` (12 tasks, all pass GATE A: stub fails held-out, gold passes
visible+held-out and is pyrefly-clean, and a type_wrong sketch surfaces a real diagnostic; surface spans
TypedDict, dataclass, generics, Protocol, enum, NamedTuple, Callable, Counter). stream_agent got a
`<check/>` action + auto-check feedback + an authoring system prompt; synth_mf got `--suite authoring
--arm {none,check,feedback}` with held-out + residual-diagnostic scoring; api_agent parity. Gated so
existing suites are unchanged.

**27B smoke (3 tasks x 3 arms):**

```
arm       held_out  residual_diag  n_checks  in_tok
none        3/3        0.0           0.0       1293
check       3/3        0.0           0.0       1340
feedback    3/3        0.0           1.0       1335
```

Null, but for a NEW reason vs section 5. Not "the wrong fix is well-typed so the checker cannot be the
unique detector", but simpler: the 27B authored all three modules CORRECTLY and TYPE-CLEANLY in a single
edit, so there were no organic type errors to catch. In `check` it never elected `<check>` (0); in
`feedback` the volunteered check ran once, found nothing, and only lengthened the trajectory. A capable
model authoring a small well-specified module makes no type errors, so the checker is redundant here too.

This points the real test at a WEAKER model (organic errors -> something for the checker to catch) or
harder tasks. Now running the full 12 tasks x 3 arms on the 7B (the capability-gated test) and the 27B
(confirm the null at N=12), temp 0.

**Full 12-task result (2026-07-04):**

```
model arm       held-out  resid_diag  edits  in_tok  n_checks
27b   none      12/12     0.17        1.2    1260    0
27b   check     12/12     0.08        1.1    1291    0 (never elected)
27b   feedback  12/12     0.08        1.1    1289    1.1
7b    none       6/12     2.17        6.3    2709    0
7b    check       3/12     2.08        8.2    3439    0 (never elected)
7b    feedback    4/12     2.08       10.4    6367    10.4
```

The type checker helps at NEITHER capability tier, for the two mirror-image reasons this whole
investigation keeps surfacing. (1) The 27B does not need it: it authors all 12 modules correctly and
essentially type-clean, never elects `<check>`, and volunteered feedback finds nothing (12/12 every arm,
confirming the smoke null at N=12). (2) The 7B cannot use it: `none` (6/12) is its BEST arm; it never
elects the checker, and VOLUNTEERING diagnostics after every edit floods it into thrash (10.4 edits,
6367 tokens, 2.3x none) while cleaning nothing up, since residual type errors are FLAT at ~2.1 across all
three arms. It sees the diagnostics but cannot act on them (edit competence, same ceiling as the dispatch
7B). Caveat: temp 0 single-seed, and the 7B's per-task solves are near-disjoint across arms (real noise),
but the aggregate is unambiguous: no arm beats none, residuals do not move, feedback only adds cost.

This parallels the navigation result exactly. Both tools are redundant for the capable model (it makes no
type errors, just as it reads the type for dispatch) and unusable for the weak one (it cannot act on the
diagnostics, just as it cannot convert a resolved definition into a correct edit). So a type checker's
value is not live authoring feedback any more than it is live navigation. Across every channel we tested
(section 5 bug-fix inference, and now authoring), the checker's information does not improve the code a
capable agent writes; its value is as a GATE that keeps the committed code's types correct so the code
stays self-describing for the next self-retrieving agent. That is the reframe, now supported on both the
navigation side (types make retrieval redundant) and the authoring side (the checker does not help write
the code, only keep it honest).

## Where could a language server still beat grep+sed? semantic vs textual (subagent analysis, 2026-07-02)

grep/sed are textual; a language server is semantic (it resolves receiver types, imports/re-exports,
overloads, inheritance). A subagent mapped where semantic resolution supplies information textual search
cannot, verified against the local blobless clones at each task's base_commit. Key results:

**Our AST resolver already ran the experiment by accident.** The `SymbolResolver` behind
`candidates.json` is shallow-semantic (barely more than grep; keyed on `^\s*(def|class) NAME`), and it
mis-resolved exactly the common-method-name deps - grep's blind spot:
- django-11138 `lower` -> flagged the `lower` template filter; truth is `lookup_type.lower()`, builtin
  `str.lower`. django-11141 `add` -> flagged the `add` filter; truth is `set.add` (`def add` has **15**
  non-test defs in django, all wrong). django-11206 `rjust`, astropy-13579 `append` (`list.append`;
  `def append` has **7** defs in astropy) - same pattern. `template/defaultfilters.py` defines **58**
  collision-prone filter names (`lower/add/join/first/slice/...`), so any `.lower()/.add()` on a builtin
  textually collides there. All four "cross-file deps" that made these tasks admissible are
  textual-collision **false positives**; a type-aware go-to-definition resolves the receiver type first
  and never chases the filter. (Generalises the earlier note that django-11138's dep was a false positive
  - the pattern is systematic.)

**Two regimes where semantic beats textual:**
1. **Receiver-type / overload disambiguation** - bites constantly, changes outcome rarely. `x.foo()`
   binds to `foo` on `type(x)`; grep returns every `foo` on every class, and when the receiver is a
   builtin the real def is not in the repo at all. But a capable agent usually infers the type from the
   call site by reading, so this saves *steps, not correctness* - except for weaker models that trust the
   first hit.
2. **Indirection grep AND codenav cannot follow** - rarer, but a genuine capability gap. Factory/
   singleton/re-export bindings have no `def`/`class`: xarray `take = _dask_or_eager_func("take")` (zero
   `def take` in the repo), sympy `I = S.ImaginaryUnit`. def-grep and our `codenav defn` both return
   nothing; only a binding-aware LSP resolves them. **Implication: `codenav` shares grep's blind spot, so
   a real test of the hypothesis needs a true type-aware server (pyright/pylsp/jedi), not our AST
   resolver.**
3. **Semantic find-references / find-implementations** - sphinx `get` = **666** textual hits vs 4 `def
   get`; django `get_prep_value` = **25** overrides. Matters on dispatch-shaped bugs (edit the right
   override), a minority of tasks.

**Honest read:** the precision advantage is real but concentrated - it changes *outcomes* mainly for
weaker agents and on indirection/dispatch-shaped tasks; for a strong agent on a localized fix it changes
path length, not correctness (astropy-14182's grep-only arm patched a `QTable` task with zero codenav).
This *tightens* the thesis rather than contradicting it: LSP precision, like LSP information, is largely
redundant for a strong agent; the efficiency/policy value is what survives.

**Proposed next experiment.** Select 10-15 tasks for textual ambiguity **at the fix site** (receiver-type
collisions like `add`/`append`/`rjust`; indirection like `take`), not just cross-file cost-gap. Two arms
on a strong scaffold with test feedback: **G** grep/sed only vs **T** grep/sed + a genuinely type-aware
goto/refs (pyright/jedi-backed, NOT the AST `codenav`). Metrics: resolved@1 (F2P) and a **mis-localization
rate** (did the patch edit the wrong same-named symbol/override, vs the gold hunks); steps/tokens
secondary. Include a **weaker model**, since the effect is predicted capability-gated. Predicted: G ~ T on
resolved@1 for a strong model (T lower on mis-localization/steps); T > G only for the weaker model or the
indirection subset. Confounds to control: test feedback masks mis-localization (a wrong edit just fails
and retries, so precision shows up as fewer loops unless the budget is tight), and selection must be at
the fix site or you measure resolver artifacts (as our scanner did). A null (T does not beat G on
correctness for a capable agent) is itself the publishable result.

## Honest next steps (for discussion / next session)

1. **Hand-audit the top ~15** from `candidates.json`. The scanner is a ranked shortlist, not the final
   word; a human confirms the fix truly needs the cross-file symbol and that the task is clean.
2. **S4/S5 environment setup.** To actually run the matrix we need to execute each task's tests, which
   needs the repo's dependencies. The plan is the official SWE-bench Docker images (docker is
   available). This is the known-heavy step; the selection scan runs first precisely so we only pay env
   cost for already-admissible tasks. Not started.
3. **Matrix runner.** Adapt the `api_agent.py` rollout loop to drive `RealRepoEnv` under the three tool
   conditions (R read-only / D +defn / I +check_types), recording success (F2P pass) and input tokens.
   The env interface matches `MultiFileEnv` closely, so this is a moderate adaptation, not a rewrite.
   Not started (deferred until the env step and the audited task set exist).
4. **Contamination.** Per the agreed constraint, we do not claim a "post-training-cutoff" benefit for
   SWE-bench Verified. If we want a contamination-controlled claim, validate it **per model** in our
   matrix explicitly first.

## Notes

- Codex (Ian's suggestion) was tried for the build and was unreliable in this environment: in read-only
  mode it web-searched to answer a local file count, and in workspace-write mode it hung past two
  minutes on a trivial task. Kimi has no CLI installed. Per the agreed fallback, the build was done solo.
- Cloned repos live under `runs/realbench/repos/` (gitignored, large). The scan JSONs and
  `candidates.json` are small and committed.

## Checker-gate v3 build: expanded cohort + phase-gradient dry run (2026-07-19)

Apparatus only — no model runs, CPU only, nothing under runs/ modified (new files added).

**Expanded seeded cohort.** `checker_hidden.py --mutation-set gate-v3 --include-clean` builds
`checker-gate-v3`: 12 defect/clean pairs (24 workspaces), one guarded mutation per authoring task,
spanning 8 checker-detectable defect families with at most 2 per family: wrong-return-type 2 (cart,
graph), misused-generic-container 1 (multimap), attribute-typo 2 (bank, shapes),
wrong-typeddict-dict-key 2 (machine, config), wrong-call-arity 2 (grid, fold), wrong-argument-type 1
(histogram), undefined-name 1 (tokenizer), bad-missing-import 1 (pipeline). The three
checker-gate-v2 pairs are reused byte-identically (workspace hashes match pilot3). Every pair passes
the SAME mechanical gates as v2 — defect: coherent, visible-pass, held-out-fail, exactly one
target-scoped semantic diagnostic, no syntax diagnostic; clean: exact validated gold, visible+held
pass, zero diagnostic delta — 24/24 rows PASS, one distinct pyrefly code per defect (bad-return,
bad-argument-type, missing-attribute, bad-index, unknown-name, missing-argument,
missing-module-attribute, bad-typed-dict-key). **0 candidates discarded**: all 12 authored mutations
cleared every gate on the first mechanical validation. Default `--mutation-set v1` output verified
identical to the HEAD generator (drafts+audit, latency excluded); tests 28/28 pass. Artifact:
`runs/protocol/checker_gate_v3_validation.json`.

**Phase-gradient arms + scripted dry run.** All four delivery arms run on this one cohort through
the existing revise grid: `control` (no checker), `diagnostics` (one-shot delta at revision, C26
delivery), `gate` (v6 acceptance gate at submission), `noisy` (after-every-edit volunteered
feedback, the authoring `feedback` arm). New `checker_paired.py dry-run` subcommand +
`scripts/experiments/stub_policy.py` replay deterministic action scripts through the LIVE
StreamAgent loop (char-level stub tokenizer/model, no weights loaded): the 96-cell grid (24 drafts x
4 arms) is all green under the v6 case-series validator, and 4 targeted mechanics scenarios pass —
(1) a user observation containing literal `<done/>` fires no completion; (2) a successful edit
invalidates stale passing-test state (EOS after the edit does not terminate); (3) full
reject -> repair -> retest -> resubmit -> accept cycle with two distinct model-origin completions and
exact gold restoration; (4) documented property: after rejection the gate mechanically requires a
fresh model-generated `<done/>` and has invalidated stale test state, but re-checks diagnostics only
(the retest is instructed, not enforced). Control/gate scripted prefixes are identical through the
first completion; diagnostics/noisy diverge earlier by design (C26/C29 handling, recorded in the
artifact's `arm_divergence_notes`). Artifact: `runs/protocol/checker_gate_v3_dryrun.json`.

**Launch later (local 27B, temp 0, GPU):**

```
PYTHON=.venv-streams/bin/python bash scripts/run_checker_gate_v3.sh
# equivalently:
.venv-streams/bin/python scripts/experiments/checker_paired.py revise \
  runs/protocol/checker_gate_v3_validation.json runs/pilot/checker_gate_v3_qwen35_27b_pilot.json \
  --model Qwen/Qwen3.5-27B --revision b7ca741b86de18df552fd2cc952861e04621a4bd \
  --arms control,diagnostics,gate,noisy --temperature 0 --seed 0 --seeds 1 \
  --max-new 1400 --max-turns 12 --max-reads 4 --gpu-only
.venv-streams/bin/python scripts/analysis/analyze_checker_paired.py \
  --drafts runs/protocol/checker_gate_v3_validation.json \
  --revisions runs/pilot/checker_gate_v3_qwen35_27b_pilot.json
```

### Exp 1 follow-up: L3 hidden-type dispatch (closes the "type always visible" caveat) (2026-07-20)

The L0/L1/L2 ladder always SHOWS app.py + the test, so the receiver type is readable in the prompt at
every rung (design caveat, ~line 481). This is the variant the log itself proposed: HIDE app.py AND the
test source, so the receiver type must be RETRIEVED (read app.py, read the test, grep, or one `<defn>`
at the given use-site) rather than read off the prompt. Same 15 annotated (L0) repos, same 27B
(Qwen3.6-27B rev 6a9e13bd, temp 0, one rollout/task), same three arms. Gated behind a new
`--visibility hidden` flag (default `visible` is byte-identical; existing suites unchanged). The prompt
keeps the behavioral spec as the test's assert lines with the receiver construction redacted to `x`
(AST-mechanical), keeps the use-site line/col as metadata so `<defn>` stays callable without a prior
read, and passes a mechanical leak guard (buggy class name + buggy file path absent from every prompt).
Artifact `runs/realbench/dispatch/local_Qwen3.6-27B_hidden.json` (45 rows; run split across two
sequential temp-0 processes after the first was killed externally at 18/45 — the reran record_to_dict
grep_base cell reproduced byte-for-byte, confirming determinism; merge note + shard paths in the file).

```
                resolved  mean_in_toks(res)  n_grep  n_defn  read_whole   fails
grep_base       12/15      1390              2.9     0.0     2.1          record_to_dict, job_priority, resource_cost
defn_avail      13/15      1454              1.7     0.4     2.0          field_validate, job_priority
defn_prompt     14/15      1349              0.7     1.0     0.9          job_priority
```

**(a) How much is readable type info worth?** The cost shows up as RESOLUTION, not tokens. grep_base
falls 15/15 (L0 visible) -> 12/15 (hidden); on the 12 tasks it solves in BOTH regimes the token cost is
essentially flat (hidden/visible = 1.016, +21 toks) and its action mix flips from "read the type, go
straight to the class" (visible n_grep 0.3, ~1 read) to a real retrieval hunt (hidden n_grep 2.9, 2.1
whole reads). So a readable receiver type is worth ~3/15 of grep_base's resolution plus the conversion
of a multi-grep hunt into a direct lookup — it does NOT raise the per-solve token cost, because when
blind grep does localize, it localizes about as cheaply as reading the annotation.

**(b) Does `<defn>` finally beat text once retrieval is required?** On RESOLUTION, yes — the first
dispatch regime in this whole investigation where it does: defn_prompt 14/15 > defn_avail 13/15 >
grep_base 12/15. Two concrete rescues (record_to_dict, resource_cost): grep_base thrashes on 9-10 `def
NAME` candidates without ever editing; with the tool the model emits one `<defn sym=... file=app.py
line=L col=C/>` at the use-site, gets the exact binding class, and solves cleanly (e.g. resource_cost
defn_avail: 0 grep, 1 defn, correct storage-charge fix). On TOKENS among matched successes `<defn>` is
still only ~neutral-to-slightly-cheaper (defn_avail ratio 1.009, defn_prompt 1.051), so the tool's win
here is localization, not efficiency. This is the report's "for a weak model the LSP's value looks like
resolution" — now shown for the CAPABLE 27B, but only once readable source is artificially removed.

**(c) Does accuracy drop?** grep_base drops 3 (above). One task, job_priority, fails in ALL three arms:
in defn_prompt the tool resolved the RIGHT class (DeadlineJob in pkg/jobs/basic.py) and the model edited
it — but mis-reasoned the comparison flip (`> 24` -> `>= 24` instead of `< 24`), so it is an
edit-competence failure, not a localization one (localization != correctness, again). defn_avail newly
loses field_validate to a different failure mode hiding introduces: with the test redacted the model
spends its budget hunting for the test file (9 greps, 0 edits) instead of reading app.py or electing
`<defn>`; defn_prompt's stronger framing avoids that and loses nothing beyond the universal
job_priority.

**Read.** Closing the caveat does NOT overturn the reframe, it bounds it precisely. When the receiver
type sits in readable source (every real repo), the capable model reads it and the LSP is redundant on
both tokens and resolution (L0-L2). The ONE regime where go-to-definition earns its keep is when
readable source is forcibly hidden — an artificial setup — and even there the payoff is a modest
resolution assist (~+2/15, driven by two localization rescues), not the synthetic 3.5-4.7x token
efficiency, because a solved retrieval is about as cheap by read as by defn. The LSP's ceiling stays
small: its best case is rescuing a localization the model would otherwise thrash, and that only appears
off the readable-source path real code never leaves.

### Exp 2: reread-after-span — substitution is not promptable-away (2026-07-20)

Context (C24, C27): in the navigation-v2 pilot every AUTOMATICALLY delivered definition span was
followed by a read of the target file (the span did not substitute for the read), while in the
retrieval suite an ELECTED definition was never re-read. Question at larger n: is the auto-span reread
persistent under an explicit sufficiency instruction (=> substitution needs training) or promptable-away
(=> prompting suffices)? Driver `scripts/experiments/run_navigation_reread.py` (a SEPARATE driver; the
frozen navigation protocol sources are imported read-only and unchanged; the reserved confirmation split
is refused by construction). Qwen3.6-27B rev 6a9e13bd, temp 0, one rollout/task, the 12 apparatus-audit
instances (NOT the 12 reserved confirmation instances — C15). Three arms, typed variant:
`auto_neutral` (auto span, neutral framing = the pilot cell), `auto_sufficient` (auto span + explicit
"the span is the complete definition; do not open the defining file unless it is insufficient"),
`framed_elective` (model must call `<defn>` itself, strong framing). Artifact
`runs/pilot/navigation_v2_reread_qwen36_27b_apparatus.json` (36 rows).

```
arm              pass    reread_target   mean_in_toks
auto_neutral     11/12   11/12           1157
auto_sufficient  10/12   12/12           1452
framed_elective  11/12    0/12 (vacuous) 1640
paired auto_neutral -> auto_sufficient (n=12): reread persisted 11, removed by instruction 0, induced 1
paired auto tokens (resolved in both, n=10): neutral 1157 -> sufficient 1429 (+272)
```

**Answer: persistent, and prompting makes it WORSE, not better.** The explicit sufficiency instruction
removes the reread in ZERO of 12 tasks; auto_sufficient rereads the defining file 12/12 (vs 11/12
neutral — the one neutral non-reread, 29089, flips TO a reread under the instruction). The instruction
also costs +272 tokens/task on matched successes (it adds prompt text and the model reads anyway) and
nudges pass@1 down (10/12 vs 11/12). So the model that receives an LSP result still issues the file
read, and telling it the span is sufficient does not stop it — this is evidence that getting SUBSTITUTION
(span replaces read) requires TRAINING, not prompting. It aligns with the report arc: election was a
trainable policy (DAgger), and substitution looks the same — a behavior the base model won't adopt from
an instruction.

**Elective arm is vacuous for substitution (honest caveat).** The strongly-framed 27B does try to elect
`<defn>` (9/12 emit one), but under the strict use-site env its elected lookup resolves a usable span in
0/12 — the model does not supply a valid file/line/col use-site, so the server returns nothing. It then
localizes by reading `pkg/factory.py` (the registry) + the target unit and edits: target file read
12/12. So "reread 0/12" for this arm is NOT substitution — there is no span to substitute; it is the
C24 "framed election rarely produces a usable result" behavior, and the target file is read anyway. Net
across ALL three arms: the defining/target file is read every time (auto: after the delivered span;
elective: because the elected tool returns nothing). The 27B never substitutes a span for the read.

Caveat: temp 0 single-seed, 12 apparatus instances (not confirmation), one capable local model;
this measures a behavioral tendency (reread-after-span and its non-response to an instruction), not a
population equivalence. The reserved 12 confirmation instances remain unconsumed (mechanically
re-validated 12/12 this session before the run).

**Exp 2b: strong-model validation (OpenRouter, tool-calling api_agent path).** Ran arms auto_neutral
and auto_sufficient of the same 12 apparatus instances through `scripts/experiments/api_reread.py`
(reuses `scripts/api_agent.py`'s key/pricing/budget guard + tool-executing Rollout; navigation tasks
imported read-only; tools = read_file/edit_lines/run_tests/done, no defn — the span is auto-delivered as
in the local automatic arms). temp 0, seed 0, budget guard on. Artifacts
`runs/pilot/navigation_v2_reread_api_{sonnet45,deepseek}_apparatus.json`.

```
claude-sonnet-4.5   auto_neutral     pass 12/12   reread_target 12/12
                    auto_sufficient  pass 12/12   reread_target 12/12     spend $0.9363
deepseek-v3.1       auto_neutral     pass 12/12   reread_target 12/12
                    auto_sufficient  pass 12/12   reread_target 10/12     spend $0.0639
```

Sonnet rereads the auto-delivered span in 24/24 rollouts — 100%, INCLUDING all 12 auto_sufficient
rollouts under the explicit "do not open the defining file" instruction (even 29089, the one task where
the local 27B substituted, sonnet rereads). deepseek-v3.1 rereads 12/12 neutral and 10/12 under the
instruction — so the instruction shaves off at most 2/12 for the weaker API model and leaves it the
majority behavior. Across all three tested models (27B, sonnet, deepseek) the picture is the same:
neutral-arm reread is ~universal (11-12/12) and the sufficiency instruction does NOT reliably stop it
(sonnet 0 removed, 27B 0 removed, deepseek 2 removed). So reread-after-span is not a weak-model artifact
and is not promptable-away for a frontier tool-calling agent: the agent opens the file to confirm before
editing regardless of being told the supplied span is complete. This is the cross-capability
confirmation that substitution is a trained behavior, not a prompted one. Total API spend $1.00,
tracked by the budget guard; both runs within budget.

### Checker-gate v3: phase-gradient run — 12 pairs x 4 delivery phases (2026-07-20)

The build's launch command, run as specified: `checker_paired.py revise` over the 12 defect/clean
`checker-gate-v3` pairs, arms `control,diagnostics,gate,noisy`, Qwen3.5-27B rev `b7ca741b…`, temp 0,
seed 0, one rollout/cell, `--max-new 1400 --max-turns 12 --max-reads 4 --gpu-only`. 96 cells (24
workspaces x 4 arms), ~58 min wall. Artifact `runs/pilot/checker_gate_v3_qwen35_27b_pilot.json`
(96 rows; run split into 4 sequential temp-0 shards of 3 pairs each so an external kill could not lose
the grid — the driver has no checkpointing and publishes atomically; cells are independent, each shard
was v6-validated on publication and the merged grid re-validated; merge note + shard paths/sha256 in
the artifact's `merge_note`). Env deviation: `HF_HUB_CACHE` pointed at the local snapshot instead of
the NAS mirror, `HF_HUB_OFFLINE=1` (known hub-metadata hang); same pinned revision, identical weights.
`nvidia-smi` is broken on this host (NVML/driver version mismatch) but torch sees the GB10 fine.

```
DEFECT cohort (n=12)     acc_clean_correct  held  type_clean  accepted_bad  gate_rej  rej>acc  tok  turns
control                        1/12         1/12    2/12         11/12          -        -     787   1.5
diagnostics (at revision)     10/12        10/12   11/12          2/12          -        -    1210   3.0
gate (at submission)          11/12        11/12   12/12          1/12        10/12    10/10   1380   4.6
noisy (after every edit)       1/12         1/12    1/12         11/12          -        -     754   1.2

CLEAN cohort (n=12)      acc_clean_correct  held  type_clean  false_rej  first_pass_acc  tok  turns
control                       12/12        12/12   12/12          -           -          591   1.0
diagnostics                   12/12        12/12   12/12          -           -          619   1.0
gate                          12/12        12/12   12/12         0/12       12/12        591   1.0
noisy                         12/12        12/12   12/12          -           -          651   1.0

task-bootstrap contrasts over all 24 workspaces (accepted_type_clean_correct):
  diagnostics - control  +0.375 [+0.250,+0.500]
  gate        - control  +0.417 [+0.292,+0.500]
  noisy       - control  +0.000 [-0.125,+0.125]
```

**(1) Does the phase gradient hold on identical defects?** Yes, monotonically: 1/12 (no checker) ->
1/12 (in-loop) -> 10/12 (revision) -> 11/12 (submission). Later delivery is better, and the whole
gradient is a step function at "delivered at all, but not in-loop": the two late arms are 10-11/12 and
the two early arms are 1/12. Gate > diagnostics by exactly one task (`auth_pipeline_handler`, where
diagnostics hands the model the delta and it submits anyway with 0 edits; the gate blocks the same
submission and it then repairs). The mechanism separating gate from diagnostics is not better
information — it is the same information plus a refusal to accept.

**(2) Does the noisy in-loop arm reproduce the authoring harm?** It does not measurably harm, but it
delivers nothing: noisy - control = +0.000 [-0.125,+0.125], and per task it is a 1-for-1 swap (control
solves `auth_multimap_generic` and noisy does not; noisy solves `auth_shapes_protocol` and control does
not). The reason is degeneracy, not feedback quality — the after-every-edit channel fires in 1/12
defect rows because the model edits in only 1/12 rows (mean edits before first submission: control
0.17, noisy 0.08). Under the volunteered-feedback system-prompt advert the 27B mostly reads, tests, and
submits the frozen draft unchanged. So at revision-time the authoring experiment's in-loop harm does
not reproduce as harm; it reproduces as a null with the feedback channel almost never exercised. The
noisy arm diverges from control at the system prompt by design (`arm_divergence_notes`), so it is not
prefix-matched to control the way gate is.

**(3) Does gate rejection precision hold at n=12?** Yes. 0/12 false rejections on the matched clean
controls; all 12 clean drafts gate-checked and accepted on the first submission, 12/12 accepted
type-clean and held-out correct, 0 accepted-dirty. On the defect side 10/12 first submissions were
rejected and 10/10 completed reject -> repair -> retest -> resubmit -> accept, type-clean and held-out
correct. The other 2/12 defects (`auth_multimap_generic`, `auth_shapes_protocol`) self-repaired before
submission and passed the gate first try, so there was nothing to reject.

**The gate's one miss is a blind spot, not a precision failure.** `auth_shapes_protocol` gate: the
model self-repairs before submitting, the repair removes the seeded type error (final diagnostics
empty), the gate sees 0 diagnostics and accepts — but the held-out test still fails. A type-clean
acceptance gate is exactly as behaviorally blind as the checker behind it; it stops defects the checker
can name and passes a wrong-but-well-typed edit. Diagnostics loses the same task the same way, plus
`auth_pipeline_handler` (delta delivered, 0 edits, submitted dirty).

**(4) Costs.** On defects, revision tokens 787 control -> 1210 diagnostics (+54%) -> 1380 gate (+75%);
turns 1.5 -> 3.0 -> 4.6. The gate's premium buys +10/12 accepted-clean-and-correct, so cost per
accepted-correct submission falls sharply (control 787 tok/(1/12) vs gate 1380 tok/(11/12)). On clean
drafts the gate is free: 591 tokens in both control and gate, byte-identical trajectories, the only
cost being one checker invocation (+135 ms mean checker latency, 122 -> 257 ms). Diagnostics pays its
one-shot delta on every draft including the 12 clean ones (619 vs 591 tok, +233 ms latency) because it
is delivered unconditionally; the gate pays only where a submission is actually attempted. That is the
deployment asymmetry: revision-time delivery taxes clean work, submission-time delivery does not.

**v6 integrity.** 106 `done_attempt` events, 0 non-model-origin; 0 rows whose first completion is not
model-generated; control/gate identical first-completion prefixes 24/24 pairs (gate pairing audit
`valid: true`, 0 invalid); 0 serialization failures; gate invoked in 24/24 gate rows; all 96 cells
terminate `done` (no truncation, no unsubmitted rows). Merged grid re-validated by the driver's own v6
case-series validator: PASS.

Caveats unchanged from C29 and now at n=12: constructed single-diagnostic defects, one model, temp 0,
one seed, 8 defect families, draft-generation cost excluded. This is a within-design phase gradient on
identical seeded defects, not a natural-prevalence or population claim.

### Exp 2b: substitution TRAINING — the reread that prompting could not remove, training removes (2026-07-20)

Direct follow-up to the reread null above (C31). Prompting removed the post-span reread on 0/12 for the
27B and 2/36 across three models. Election was a trainable policy on a 7B (C2, `run_relabel2.sh`); this
run asks whether SUBSTITUTION is trainable on the SAME model where the null was measured.
Qwen3.6-27B rev `6a9e13bd`, local GPU only, no API calls.

**Apparatus.** New driver `scripts/experiments/run_substitution_train.py` (harvest) +
`scripts/run_substitution_train.sh` (harvest -> LoRA SFT -> retest) +
`scripts/analysis/analyze_substitution.py`. Structure mirrors `run_relabel2.sh` exactly: the model
rolls out with the span auto-delivered, chooses `<read>` of the span's own file, a rule oracle DENIES
the read and states the span is complete, and the read-attempt+redirect prefix is DROPPED so the
model's own next action is the first trained action from a CLEAN prompt (the relabel2 fix). Only the
span's file is denied, only before the first applied edit; grep and other reads stay available.

`scaffold/stream_agent.py` is NOT modified — it is hash-gated by `run_navigation_confirmation.sh`, so
the oracle patches the env instance and the relabel trace is mirrored bit-for-bit by wrapping the
agent's `_ids`/`_prefill`. The mirror is asserted equal to the agent's own decoded stream before any
row is kept: **48/48 rows verified**. Additive, default-off flags only elsewhere: `sft_lora.py` gains
`--revision`/`--filter sft_keep`, `run_navigation_reread.py` gains `--adapter` (inert when absent).

**Splits (disjoint by construction).** Training = a FRESH `substrain` split, 12 instances, seeds
53003-53101, templates `add`/`multiply` — disjoint seeds AND templates from the apparatus retest set
(subtract/affine/xor) and from the reserved confirmation set, which is never built and whose seeds
appear in no artifact of this run. Full navigation-v2 mechanical validation **12/12 PASS**
(`runs/protocol/navigation_v2_substrain_validation.json`). Retest = the 12 HELD-OUT apparatus
instances, the exact set where the reread null was measured. Harvest/retest overlap: NONE.

**Harvest** (48 rollouts = 12 tasks x 4 seeds, temp 0.7): held-out pass 46/48, **redirect fired 46/48**
(the model attempts the reread in 96% of rollouts — the C31 behavior replicated on fresh templates at
temp 0.7), 39/48 clean substitution demos kept (mean 190 trained tokens each; rejects are mostly
retrieval reappearing inside the kept segment, 12, plus 2 held-out failures and 2 spontaneous
substitutions with no redirect to drop). Artifact `runs/agent/substitution_harvest_qwen36_27b.json`.

**SFT.** LoRA r16/alpha32 on q,k,v,o,gate,up,down; 79.7M trainable (0.295% of 27.0B); 39 examples,
3 epochs, lr 1e-4, bs 1 x accum 4, 117 micro-steps; loss 0.39 -> 0.15. Adapter
`runs/sft/substitution_lora_27b`.

**Retest** on the 12 held-out apparatus instances, `auto_neutral` arm, temp 0, seed 0.

```
arm                        pass    read_after_span   mean_in   mean_total   mean_reads
baseline (untrained)       11/12    11/12             1157      1543         3.42
substitution-trained       10/12     0/12              748      1030         0.08
paired (n=12): removed 11, persisted 0, induced 0, absent 1
tokens on the 9 instances resolved in BOTH arms: 1493 -> 938 (base/trained 1.591, -555/task)
for contrast, C31 prompting: removed 0/12 for this model, +272 tokens/task
```

**Answer: trainable.** The behavior that an explicit "the span is the complete definition; do not open
the file" instruction could not shift at all — 0/12 removed, and it made tokens and pass@1 worse — is
removed on 11/11 instances where it occurred by 39 demonstrations of the model's OWN post-redirect
action, generalizing across disjoint seeds and disjoint task templates. Total agent reads fall 3.42 ->
0.08/task and matched-success tokens fall 37%. This closes the loop the report's arc predicted:
election was a trainable policy, and substitution is the same kind of object — a behavior the base
model will not adopt from an instruction but adopts readily from on-policy demonstrations.

**Untrained control (determinism).** The untrained baseline was re-run through the modified driver on
the same 12 instances: pass 11/12, read_after_span 11/12, mean_in 1157.0 — **byte-identical** to the
frozen C31 artifact on every outcome, token and stream field for all 12 rows
(`runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_baseline_rerun.json`). The change is
attributable to the adapter, not to driver drift.

**Correctness caveat (honest).** Held-out pass moves 11/12 -> 10/12: one rescue (29021, which the
baseline failed) and two losses (29033, 29077, both `xor` tasks). All three trained failures/successes
edit the CORRECT file — `wrong_file_edits` is 0/12 and first-edit path is right 12/12 — so localization
is intact; the two losses pass the visible test and fail the held-out oracle, i.e. partial-spec
overfit on the constant. The adapter also compressed deliberation to an empty `<think>` block before
editing, which plausibly drives the extra arithmetic slips. At n=12 and one seed this -1 is not
distinguishable from noise, but it is the direction to watch: substitution buys ~37% of tokens and may
cost a little checking.

**Scope.** One model, one seed at retest, one task family (navigation-v2 dispatch), 12 held-out
instances, and an oracle that denies only the span's own file. This shows substitution is trainable
for this model on this family — not that a substitution-trained agent is safe to deploy, and not that
the token saving survives on tasks where the span is genuinely insufficient (no read-required boundary
set was included, unlike the relabel2 boundary arm).

Artifacts: `runs/protocol/navigation_v2_substrain_validation.json`,
`runs/agent/substitution_harvest_qwen36_27b.json`, `runs/sft/substitution_lora_27b/`,
`runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_trained.json`,
`runs/pilot/navigation_v2_reread_qwen36_27b_apparatus_baseline_rerun.json`.

## Push-vs-elect reread: election is not substitution (2026-07-24)

Clean within-instance follow-up to C31/C33. Question: does a SELF-ELECTED span (vs a pushed one) get
substituted? C27 saw 0/11 reread when a 27B elected in the `effic_real2` suite; C31's own elective arm was
vacuous (composed resolver returned a usable span 0/12). This run fixes that — same navigation apparatus
instances, two arms (`push_span` / `elect_span`), the identical span mechanically verified usable and
byte-identical across arms (sha match 12/12, 7-point usability 24/24), delivered in `elect_span` only on
the model's own tool call.

Qwen3.6-27B (local, temp 0), GLM-4.6, GPT-5.6 "Luna" (both OpenRouter), 12 instances x 2 arms each. Spend
~$0.35.

```
model            push reread   elect reread (among elected)   elected/12
Qwen3.6-27B        11/12         9/9                            9
GLM-4.6            12/12         4/9                            9
GPT-5.6 Luna       12/12         7/12                           12
```

Findings: (1) push -> reread is universal (35/36), replicating C31. (2) Election does NOT reliably produce
substitution: the local 27B rereads on every instance it elected (9/9, zero effect), and the two frontier
API models reduce but do not eliminate the reread (4/9, 7/12). (3) So C27's clean elective substitution is
specific to that suite/model, not a general property. (4) Training remains the only reliable lever (C33:
0/12). Caveats: the local arm is the clean within-instance contrast (defn delivery recorded 9/9); the API
`elect_span` arm delivered an identical usable span (payload present) but records thinner counters
(defn_found/n_lsp null on the API path); GLM-4.6 solved only 5-6/12 of the harder xor/arithmetic tasks
(reread measured independently of success); n=12, one seed, single apparatus. Artifacts:
`runs/pilot/navigation_v2_reread_pushvselect_{qwen36_27b,glm46,gpt56luna}.json`.
