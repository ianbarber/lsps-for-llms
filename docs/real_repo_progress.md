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
