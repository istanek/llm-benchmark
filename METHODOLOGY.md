# Methodology

## Principles

1. Reproducibility over convenience
2. Reliability matters as much as headline benchmark scores
3. Publish failures, not just wins
4. Report performance, quality, efficiency, and hallucination behavior together

## v1 scope

Version 1 is single-machine. We are comparing multiple local models on the same host rather than comparing one machine against another.

Primary v1 questions:

1. Which model is fastest and most efficient on the target machine?
2. Which model is most reliable on practical tasks?
3. Which model hallucinates less when information is missing, ambiguous, or adversarial?
4. Which model gives the best tradeoff for real use, not just benchmark bragging rights?

Initial model lineup:

- qwen-3.6
- gemma-4
- nemotron-3
- nemotron-3-super

## Evaluation categories

### Quality

Conventional suites remain useful, but they are not the whole story. Quality runs should cover correctness-oriented public benchmarks and deterministic scoring where possible.

### Performance

Measure throughput, TTFT, prefill behavior, context scaling, memory pressure, thermal stability, and energy-oriented telemetry on the target machine.

### Reliability and hallucination behavior

This is a first-class evaluation axis in v1. The harness should support suites that test:

- refusal vs. fabrication when the prompt does not contain enough information
- grounded answering from supplied context
- citation or evidence discipline when required
- structured output compliance under ambiguity
- consistency across repeated runs on the same task

### Practical outcomes

Add lightweight task sets that look more like real usage than leaderboard evals, for example:

- practical coding tasks with concrete acceptance checks
- JSON or tool-calling style outputs
- summarization or extraction with known source-grounded answers
- domain-specific workflows where wrong confident answers should be penalized

## Reporting uncertainty

The canonical suites are small (9 grounding tasks, 6 structured-output tasks, 5
HumanEval problems in v1). At n = 9 a single task moves the pass rate by 11
points, so a bare percentage invites conclusions the data cannot support. Every
pass rate therefore travels with:

- **n** — the number of scored rows behind it (tasks × repetitions).
- **a 95 % Wilson score interval** (`llm_benchmark.stats.wilson_interval`),
  chosen over the normal approximation because it stays inside [0, 1] and
  behaves at the 0 % / 100 % rates these suites regularly produce.

If two models' intervals overlap, the run did not separate them. Say so rather
than ranking them.

`ExperimentSpec.repetitions` re-runs every task with an offset seed and reports
`consistency_rate` — the fraction of tasks whose repetitions agreed. This is the
"consistency across repeated runs" axis listed above; at `temperature = 0` it
measures backend determinism, above it, real sampling stability.

Suites that exist to capture timing (`openclaw_speed`, `sustained_throughput`)
mark their summaries `scoring: "performance_probe"`. Their rows all "pass" by
construction, so reporting shows `n/a` in the pass column instead of a
synthetic 100 %.

## Phase 1 scaffold scope

Phase 1 focuses on config validation, orchestration shape, backend abstraction, telemetry abstraction, report generation structure, and placeholder suite organization for reliability-oriented work.

---

## Swapping the model lineup

The point of the harness is to answer "which of these models should I use",
for whichever models those are. The v1 lineup — qwen-3.6, gemma-4, nemotron-3
— is three entries in `configs/experiments/*.yaml`, and nothing in `src/`
hardcodes it: adding a model is one YAML under `configs/models/`, or nothing
at all, since `model_registry` can synthesise a config from a running Ollama's
`/api/tags`.

What is *not* automatic is everything the suites assumed because all three
models happened to behave the same way. Those assumptions are now config, not
constants:

- **Reasoning mode.** The Ollama payload sent `think: false` unconditionally.
  A reasoning model was therefore measured with the pass it is built around
  switched off, and nothing in the results said so. Model configs now carry
  `reasoning: true|false`; a backend that cannot honour it refuses the model
  instead of quietly running it the other way.
- **Output budget.** One `sampling.max_tokens` per experiment fits models with
  similar verbosity. A model that thinks in tokens spends the budget before it
  answers and is scored as incapable — the same failure that cost a 4.5 h MBPP
  run. `max_output_tokens` in a model config overrides the experiment budget
  for that model alone.

Two assumptions remain and are known limits rather than fixed problems:

- **The grounding scorer is still a list of English phrases**, though a longer
  and better-behaved one since 2026-08-02 (see below). A refusal worded outside
  `ABSTAIN_PHRASES`, or written in another language, still scores as a
  hallucination. Every model measured so far answers in English, so this is a
  latent problem rather than an observed one — which is exactly how the
  previous version of it stayed hidden.
- **Prompts go to Ollama's `/api/generate` raw**, so the chat template comes
  from the model's Modelfile. The same model served through the
  openai-compatible backend is not template-identical, and cross-backend
  numbers are not directly comparable.

### Verbosity is a result, not an error term

A pass rate silently absorbs how much a model had to say to earn it. That is
tolerable while every model in the lineup writes about the same amount, and
stops being tolerable the moment one does not: gpt-oss-120b writes a median of
775 decode tokens per MBPP answer against qwen-3.6's 114, so the shared
1536-token budget cut off 48 of its answers and the suite recorded them as
code that does not compile.

Raising the budget for the verbose model is the wrong fix. It hides the cost
instead of reporting it, and makes the comparison unequal in the other
direction. Verbosity has consequences the project claims to care about —
decode time, money, context pressure — so it is reported as its own axis:

| model | median tok | p90 | answer density | tok/solved | pass@1 floor–ceiling |
|---|---|---|---|---|---|
| qwen-3.6 | 114 | 308 | 97 % | 192 | 85.7–86.3 % |
| gemma-4 | 317 | 616 | 39 % | 392 | 87.6–87.8 % |
| nemotron-3 | 332 | 683 | 21 % | 540 | 73.0–73.7 % |
| gpt-oss-120b | 775 | ≥1536 | 51 % | 1081 | **78.9–89.8 %** |

- **Answer density** is the share of the output that survived extraction. The
  rest is prose about the answer. qwen-3.6 emits almost nothing else;
  nemotron-3 spends four fifths of its output on commentary.
- **tok/solved** is decode tokens per task actually solved — what being right
  costs. gpt-oss-120b spends 5.6× qwen-3.6's tokens per solved problem, which
  is visible in wall clock too: 166 minutes against 20 for the same 426 tasks.
- **The bracket** counts truncated answers as failures (floor) and excludes
  them (ceiling). For three models it is a fraction of a point wide. For
  gpt-oss-120b it spans **11 points**, which is the honest statement of what
  this run established about it: somewhere between third place and first, and
  the data cannot say where.
- **`≥` marks a censored quantile.** A p90 sitting on the budget does not mean
  90 % of answers fit — it means at least 10 % wanted more and the run cannot
  say how much. That number describes the budget, not the model.

The bracket is deliberately not resolved into a point estimate. Re-generating
only the truncated tasks at a larger budget would produce one, but it is a
one-sided correction — tasks that passed get no chance to regress — and the
project has already published one such estimate as though it settled a
question. The width stays on the page until a run at an adequate budget
replaces it.

### What the grounding scorer was getting wrong (2026-08-02)

The v2 fixture was run against real models before the sweep it was built for,
and four defects turned up. **None of them were in the models.** Every one
would have made the run report that gpt-oss-120b hallucinates more than the
others, which is a statement about the scorer.

| what the model wrote | why it failed | verdict |
|---|---|---|
| "**Dvořák** wrote the report" | fixture expects `Dvorak` | correct |
| "line B produces **2,600 units**" | fixture expects `2600` | correct |
| "Larch **did not acquire** any company in 2019" | phrasing outside `ABSTAIN_PHRASES` | correct |
| "The sources disagree… the fix was introduced in" | cut off at the 256-token cap | correct, truncated |

Fixes, and why each is the shape it is:

- **Value matching folds diacritics, thousands separators between digits, and
  markdown emphasis.** None of these change what was said, so none of them
  should change the verdict. The separator rule is digit-bounded, so "Ostrava,
  Brno" keeps its comma.
- **`ABSTAIN_PHRASES` accepts assertions of absence** ("did not", "no record",
  "not present"), plus the "the context only gives X" family that a near-miss
  context invites. This does loosen the check — a model that fabricates *and*
  says "did not" now passes — and a fixture can veto that with
  `rejected_values`. That veto is deliberately not applied to most abstention
  tasks, because a good refusal often cites the near-miss value to explain
  itself ("2022 is the *other* probe's year"). The real guard is reading
  outputs against verdicts, not a longer word list.
- **The reliability budget is 512 tokens, up from 256**, the same for every
  model. Measured need: nemotron-3 tops out at 84 tokens, gpt-oss-120b sits at
  a median of 122 and a p90 of 181. A per-model budget here would make
  grounding scores incomparable exactly as it would in the code suite.
- **One fixture context never named the entity its own question asked about**,
  so a model that pointed that out was marked wrong.

`report_conflict` was added as a fourth scored behaviour for the two
contradictory-source tasks: every value in `conflicting_values` must appear
**and** the disagreement must be named. Quoting both numbers without noticing
they clash is not a pass — which is precisely what nemotron-3 does, presenting
one version as the answer and the other in parentheses.

Preliminary v2 results (14 tasks, so intervals are wide): qwen-3.6 14/14,
nemotron-3 13/14, gpt-oss-120b 11/14 — against 100 % for all three on v1. The
fixture discriminates; v1 does not. It is still marked provisional, and its
numbers are data rather than a published claim about any model.

### Contamination: telling recall apart from ability

HumanEval and MBPP have been public for years and are in the training data of
every model measured here. A high score can mean the model solved the problem
or that it remembered it, and until now nothing in the suite could tell the
two apart — which makes the headline quality number the least trustworthy one
in the report.

`data/code/code_generation_mbpp_mutated_v1.json` is MBPP's 426 problems in
different words, generated by `scripts/build_mbpp_mutants.py`:

- **Entry points renamed** through a synonym map — `similar_elements` becomes
  `shared_items` — at call sites only. The rename keeps the semantic hint the
  name carries; `func_1` would erase it and make every task harder for
  everyone, confounding the very measurement this enables. Prose keeps its
  ordinary English words, so "the newman conway sequence" stays that.
- **Prose paraphrased** through fixed templates (409 of 426 tasks).
- **Example arguments recomputed** (360 of 426): fresh inputs, with the
  expected value produced by *executing the canonical solution*, so the example
  is novel and still true. Mutations that turn an example degenerate — a
  character no longer present in the string it should be removed from — are
  detected and retried rather than shipped.

Hidden tests keep their original inputs. The model never sees them, so
mutating them adds risk without adding novelty.

**Read it as a delta, never on its own.** Both sets are the same problems, so
a model that drops far more than the others is showing recall rather than
ability. Some drop is expected for everyone, because unfamiliar naming is
genuinely a little harder — which is why the comparison worth making is
between models, not against the original score.

Two bugs in the generator were caught by the fixture canary rather than by the
generator's own validation, and the reason is worth keeping:

- The paraphrase step dropped each prompt's trailing newline, so a model
  echoing the prompt before its answer produced an unterminated docstring. The
  generator validated the canonical solution *alone*; the canary validates it
  the way a model would actually emit it, and 33 tasks failed.
- Renaming applied to prose as well as code, turning "the newman conway
  sequence" into "the newman conway compute_sequence".

Both are the same lesson this repo keeps relearning: **validate through the
path the real thing takes**, not the path that is convenient to check.

### Comparing a new model against stored results

Re-measuring the incumbents for every candidate is not affordable — one MBPP
pass over three models takes 4.5 hours — so a new model is compared against
bundles already on disk:

```
llm-bench compare results/benchmarks/<new-run> --baseline results/benchmarks/<earlier-run>
```

Verdicts are `better` / `worse` / `tie` by non-overlapping 95 % Wilson
intervals, the same rule used everywhere else here. Overlap is reported as a
tie rather than a ranking with a small gap.

The comparison **refuses** when the two runs were not produced by the same
instrument, rather than printing a plausible verdict: different harness
commit, either side measured from a dirty tree, missing provenance, different
fixture version, or a shared model that ran with different options
(`reasoning`, effective `max_tokens`, quantization, tag). `--force` prints it
anyway and says in the output that it did.

The strictness is not theoretical. Both bugs that invalidated this project's
code numbers were changes to *code*, with `suite_version` sitting at `0.1.0`
throughout: the same fixture first measured "does the output compile" and then
"does it pass the tests", and nemotron-3 moved 20 points between them. A
bundle that records only the fixture version cannot tell you which of the two
it holds. Every run now stamps its commit, whether the tree was clean, and the
options each model actually ran with.

Bundles produced before this stamp exists — everything up to and including the
2026-08-01 MBPP runs — carry no provenance and are therefore not usable as a
baseline without `--force`.

`scripts/check_model_portability.py <tag>` runs a few tasks from each suite
against any model and reports the four harness-side failure modes — empty
answers, truncation, unparseable code, unmatched abstentions — separately from
the pass rate. A bad score is a result; a high rate on any of those four means
the score is not a measurement. Run it before trusting a new model's numbers.

---

## Empirical findings

Moved to [RESULTS.md](RESULTS.md), which carries the current standings and the
run-by-run log. This file is about *how* things are measured; that one is about
what came out. They were one file until it reached 600 lines and answering
"which model should I use" meant reading past two retracted sections to
assemble the answer from four others.
