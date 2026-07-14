# When Do Language Servers Help Coding Agents?

## Abstract

Language servers can help coding agents by resolving bindings that text cannot distinguish, returning
compact code context, and checking patches. Across the experiments, the recurring boundary is whether
repository semantics are the bottleneck and whether the integration changes the agent's work. A correct
definition that neither changes localization nor replaces reading, or a diagnostic the agent cannot act on,
adds cost without improving the result.

The experiments test these opportunities separately. Compact definition retrieval cut input tokens by
3.5-4.7x at unchanged success when it replaced whole-file reads; a live-first Pyrefly/AST hybrid showed the
same direction. Semantic navigation was near token-neutral when text already exposed the target. In sound
typed/erased tasks, type information let Pyrefly distinguish the correct implementation from many same-named
alternatives, but the two-task agent pilot reread every target file and gained neither correctness nor lower
cost. On two selected workspaces with checker-detectable errors, one-shot diagnostics produced one additional
type-clean result without improving held-out correctness and added 217 revision tokens. The gate comparison
did not measure defect prevention.

For practitioners, start with the cheapest workflow that resolves the task. Add typed semantic resolution
when it prevents wrong-target work, compact semantic spans when they replace broad retrieval, and diagnostics
when coherent patches contain actionable static errors. Measure changed actions and outcomes rather than
tool calls. Improve election only after demonstrating service value, and deploy gates only when
measurements show that bad submissions are actually rejected.

## Practitioner guide

| Situation | Default | Evidence in this report | Keep it when |
|---|---|---|---|
| The binding is local, visible, and unique | Text search plus ranged reads | Text is competitive when prompts expose the receiver and target ([C6](evidence/claim_ledger.md#c6), [C9](evidence/claim_ledger.md#c9)) | Localization is reliable at low cost |
| Overloads, inheritance, factories, or re-exports make the target ambiguous | Typed semantic resolution | Sound types improve resolver precision; agent-level benefit remains open ([C15](evidence/claim_ledger.md#c15), [C24](evidence/claim_ledger.md#c24)) | It prevents wrong-target work or reduces retrieval |
| A compact result can replace broad reading | Definition or enclosing-method span | 3.5-4.7x fewer input tokens against whole-file reads; the live-first result uses a composed Pyrefly/AST integration ([C1](evidence/claim_ledger.md#c1), [C4](evidence/claim_ledger.md#c4)) | The result substitutes for retrieval rather than preceding the same read |
| A coherent patch contains checker-detectable errors | Show new relevant checker errors after the patch | One additional type-clean result in two selected cases; no correctness gain ([C26](evidence/claim_ledger.md#c26)) | The model repairs the defect and held-out outcomes improve |
| A useful service is available but rarely chosen | Improve tool description or policy | Prompting and training change election in model-specific runs ([C2](evidence/claim_ledger.md#c2), [C3](evidence/claim_ledger.md#c3)) | Higher election retains correctness and lowers total cost |

**Open integrations:** gate prevention is unmeasured because the local comparison is invalid and contains no
rejection event ([C14](evidence/claim_ledger.md#c14), [C26](evidence/claim_ledger.md#c26)). Candidate
reranking, constrained generation, and structured edits are discussed in related work but have no direct
repository experiment.

## 1. How language-server assistance creates value

Language-server assistance addresses three distinct bottlenecks:

- **Resolve:** identify the correct program entity when names or local text are ambiguous. Type information,
  whether written or inferred, can distinguish overloads, implementations, and factory results.
- **Compress:** return the needed definition, signature, or method span instead of making the agent search
  and read broader source context.
- **Validate:** expose a checker-detectable defect in a coherent patch while the agent can still repair it,
  or prevent a bad patch from being accepted.

Agents can request these services, receive results automatically, or use them through patch feedback,
acceptance gates, candidate reranking, structured operations, and training rewards. Delivery changes
availability and election; it cannot make redundant information useful.

The common value chain is:

`real opportunity -> useful semantic signal -> changed agent action -> lower cost or higher correctness`

Each step must be measured. Correct resolution is not an agent benefit if the same target was already clear.
Compact context saves nothing if the agent rereads the file. A diagnostic has no outcome value on a clean
draft or when the model cannot repair the reported error.

## 2. Results

The [claim ledger](evidence/claim_ledger.md) maps every material claim to its artifacts and status. The
[manifest](evidence/manifest.json) records hashes, configurations, and provenance warnings.

### 2.1 Compact retrieval works when it replaces a whole-file read

The `effic` and `effic_real2` tasks ask an agent to implement a small target with an unfamiliar library API.
The treatment returns a compact definition; the counterfactual drives whole-file retrieval. Every attempt
in both arms succeeds.

| Model | Pooled input-token ratio | Tasks cheaper with definition | Mean task ratio (95% bootstrap CI) | Success |
|---|---:|---:|---:|---:|
| Qwen3.6-27B | 3.50x | 11/11 | 4.02 (2.85-5.30) | 44/44 both arms |
| Claude Sonnet 4.5 | 3.65x | 11/11 | 3.65 (2.72-5.20) | 44/44 both arms |
| DeepSeek v3.1 | 4.70x | 10/11 | 6.03 (2.32-10.82) | 44/44 both arms |

The task-level medians are 3.83, 3.01, and 2.70; DeepSeek's distribution is strongly skewed. Seeds are
repeated runs within tasks, and intervals resample task means. The three-model comparison uses the
repository's static AST resolver and measures compact retrieval against whole-file reading
([C1](evidence/claim_ledger.md#c1)).

Tool election is a policy question. In matched local 7B runs, definition-trained and read-trained policies
both solve 40 cells while mean input falls from 3,191 to 684 tokens. Relabeling raises definition use from
approximately zero to 100%; cost-reward training reaches 86% use with 36/36 success. Prompting and training
can change election, but these runs do not establish a model-size law
([C2](evidence/claim_ledger.md#c2), [C3](evidence/claim_ledger.md#c3)).

A live-first Pyrefly suite reduces mean input from 2,894 to 689 tokens and raises success from 14/24 to
24/24. The tool combines live lookup with AST-selected use sites, span expansion, and fallback, and the rows
do not record which backend answered. The estimate therefore applies to the composed integration
([C4](evidence/claim_ledger.md#c4)).

**Interpretation:** compact definitions reduce cost when they replace coarse retrieval. Their advantage over
efficient grep plus ranged reads remains unmeasured here.

### 2.2 Semantic navigation helps when resolution changes the work

A 15-task dispatch suite compares grep/ranged reads with live Pyrefly goto. On the annotated 27B variant,
grep, neutral goto, and framed goto solve 15/15, 14/15, and 15/15; matched-success token ratios are 0.972 and
1.041. Stripped and factory-indirection variants remain near one (0.945-1.065).

The prompts expose the concrete receiver and make the correct file cheap to identify, so this suite measures
a redundant-navigation regime. It has no erased or `Any` condition, and some variants still describe an
annotation that has been removed. Goto adds little when text already reveals the answer
([C6](evidence/claim_ledger.md#c6),
[C9](evidence/claim_ledger.md#c9)).

The typed/erased navigation pilot isolates this mechanism. Typed and erased repositories have byte-identical
runtime code, tests, and gold patches. Only `factory.pyi` differs: the typed stub has sound per-key `Literal`
overloads; the erased stub returns the base type. Each task contains 8-15 same-named overrides, and neither
the prompt nor failure names the target class or path. Strict live Pyrefly resolves the definition at the
use site without AST fallback. Mechanical checks confirm that typed lookup reaches the gold override,
erased lookup returns a non-discriminating base result, widening the key removes the discrimination, and
both variants are type-clean ([C15](evidence/claim_ledger.md#c15)).

The 7B and 14B models fail a supplied-span edit control. Qwen3.6-27B passes it, so the two-task comparison
uses that model:

| Outcome | Result |
|---|---:|
| Correctness | All 12 task-condition runs pass |
| Typed automatic / typed textual tokens | 1.037 (task bootstrap 0.988-1.093) |
| Erased automatic / erased textual tokens | 1.190 (1.119-1.251) |
| Typed-by-automatic interaction | -214 tokens (-402 to -26) |
| Automatic-result substitution | Every result is followed by a target-file read |
| Neutral / framed elective use | 0/2 and 1/2 |

The typed ratio lies inside the prespecified 0.90-1.10 margin for these two pilot tasks; two tasks are
insufficient for population equivalence. Lookup itself takes about six seconds per task, dominated by a
fixed indexing wait. Including that lookup, descriptive end-to-end times are about 109 seconds for automatic
context and 64 seconds for textual baselines. Correctness is ceilinged, and the tool does not replace
retrieval ([C24](evidence/claim_ledger.md#c24)).

**Interpretation:** sound types improve resolver precision. Agent value requires that precision to change
localization, prevent wrong-target work, or reduce retrieval; automatic delivery alone can turn a correct
answer into extra context and latency.

### 2.3 Checker feedback requires a coherent, repairable error

The broad checker suites do not isolate feedback value. Frontier inference arms and 27B authoring arms are
at ceiling (18/18 and 12/12), while the 7B authoring arms use different first drafts. Exact recovery finds
two coherent workspaces with relevant semantic diagnostics, but natural-draft calibration produces no usable
opportunity: 0/3 7B drafts are coherent, and the 2/8 coherent 14B drafts are already type-clean
([C11](evidence/claim_ledger.md#c11), [C16](evidence/claim_ledger.md#c16),
[C22](evidence/claim_ledger.md#c22)).

A paired case series forks each of the two recovered workspaces into control and one-shot diagnostic
revisions:

| Outcome | Control | One-shot diagnostics |
|---|---:|---:|
| Type-clean final workspace | 1/2 | 2/2 |
| Held-out tests pass | 1/2 | 1/2 |
| Accepted, type-clean, and held-out-correct | 1/2 | 1/2 |
| Mean revision tokens | 1,368 | 1,585 |

Diagnostics improve an intermediate checker state on one selected task, at 217 extra revision tokens, but
do not improve behavioral or joint success. The sample contains two selected workspaces and one seed;
draft-generation cost is unavailable
([C26](evidence/claim_ledger.md#c26)).

The paired gate arm diverges from control before either attempts completion and records no rejection event,
so prevention is unmeasured.

**Interpretation:** diagnostics can improve checker state without improving behavior. Their value depends on
coherent patches with actionable diagnostics, successful repair, and a better accepted outcome at acceptable
cost.

### 2.4 Execution feedback and external validity

Execution feedback is ceilinged in the committed small-task suite: two frontier models, 14 tasks, three
seeds, and three delivery modes all pass, for 252/252 attempts. This does not establish equivalence outside
small simulable functions.

The bounded real-repository scan found no fully admissible task. One Django case has a working environment
and substantial override ambiguity, but leakage and fix-site resolution were not fully audited. Both
recorded arms pass and the semantic tool is not elected. Constructed tasks therefore provide the cleanest
causal apparatus here, while population validity remains open. The audit and rejection reasons are in
[docs/external_validity_recon.md](docs/external_validity_recon.md).

## 3. Related work

Prior work studies missing context, automatic delivery, coherent drafts, and generation constraints. These
conditions map to different points in the resolve-compress-validate chain.

| Work | Relevant lesson |
|---|---|
| [Typed Holes](https://arxiv.org/abs/2409.00921) and [LSPRAG](https://arxiv.org/abs/2510.22210) | Push types, definitions, or references when context is missing; automatic delivery removes election failure. |
| [STALL+](https://arxiv.org/abs/2406.10018) | Static analysis effects vary by language and integration phase; retrieval and semantic analysis can complement each other. |
| [CoCoGen](https://arxiv.org/abs/2403.16792) | Check a coherent draft, then retrieve context to repair project/API mismatches. This is the opportunity-conditioned regime absent from after-every-edit feedback. |
| [CodeStruct](https://arxiv.org/abs/2604.05407) | Structured reads, edits, and syntax validation improve actionability through program structure rather than type information. |
| [CompCoder](https://arxiv.org/abs/2203.05132), [type-constrained generation](https://arxiv.org/abs/2504.09246), and [RLCSF v2](https://arxiv.org/abs/2510.22907) | Compiler and server signals can rank, constrain, or train candidates even when repair-time feedback is not useful. |

Across this report and prior work, static tooling helps when it supplies missing, correct, compact,
actionable information at the phase where the model can use it. The common failure modes are duplicate
information, no checker opportunity, ignored or noisy delivery, and cost that exceeds the work saved.

## 4. Reproducibility and limits

From a clean clone:

```bash
python3 -m pip install -e '.[dev,analysis]'
python3 scripts/analysis/reproduce_all.py
```

The reproducer verifies the manifest, reruns retained analyzers, recomputes task-level effects, and executes
the navigation manipulation checks without making model or API calls. Model runs are separate; see
[evidence/protocols.md](evidence/protocols.md) before using the navigation or checker run scripts.

The main limits are:

- Static tooling does not address dynamic behavior, incorrect annotations, `Any`-heavy boundaries,
  environment failures, or logic outside the checker.
- Compact-retrieval tasks are synthetic and compare against whole-file reads; efficient ranged retrieval is
  the harder baseline.
- The typed navigation outcome comes from two pilot tasks with ceilinged correctness; there is no outcome
  headroom or retrieval substitution to justify confirmation.
- The checker result comes from two selected recovered workspaces. It does not estimate natural opportunity
  prevalence or full end-to-end cost, and the gate comparison is invalid.
- The real-repository scan found no fully admissible case, so external validity is unresolved.
- Some retained artifacts contain provenance gaps, including unavailable seed shards, incomplete server
  backend/version records, source-hash mismatches, and invalidated runs. Exclusion reasons and discrepancies
  are reported in the claim ledger and manifest.

Within these limits, the evidence supports compact retrieval against whole-file reads and the mechanistic
claim that sound types sharpen semantic resolution.

## Conclusion

Language-server services can create value by resolving ambiguity, compressing retrieval, or validating a patch.
The experiments demonstrate compact retrieval against broad reads and type-driven gains in resolver
precision. They do not yet demonstrate a general navigation benefit, a checker-driven correctness gain, or
gate prevention.

The deployment standard is the same for each service: identify a real opportunity, verify that the semantic
signal changes the agent's action, and measure whether that change lowers cost or improves correctness.
Availability and invocation are not outcomes. The useful integration is the smallest targeted operation
that removes work or prevents an error the agent would otherwise carry forward.
