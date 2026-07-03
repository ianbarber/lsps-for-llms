# Real-repo experiment: overnight progress (2026-06-30 → 07-01)

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
