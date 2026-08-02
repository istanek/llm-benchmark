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

- **The grounding scorer is a list of English phrases.** A refusal worded
  outside `ABSTAIN_PHRASES`, or written in another language, scores as a
  hallucination. This is a false-negative machine for any model whose register
  differs from the v1 three.
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

Measurement provenance: the v0.4.x runs below were taken on an NVIDIA DGX
Spark (128 GB unified memory, Ollama backend) in June 2026. Hardware is named
where it changes how a number must be read, not as positioning — every
finding is a model-to-model comparison on one host.

### Scorer validation (v0.6.0, 2026-07-31)

The v0.6.0 grounding scorer was validated by re-scoring one real run
(`results/benchmarks/20260731T205220Z-93efc960`, qwen-3.6 / gemma-4 /
nemotron-3 via Ollama, 9 tasks x 3 repetitions per model) with both the old
and the new implementation:

| Model | old scorer | new scorer |
|---|---|---|
| qwen-3.6 | 21/27 = 78 % | 27/27 = 100 % |
| gemma-4 | 18/27 = 67 % | 27/27 = 100 % |
| nemotron-3 | 27/27 = 100 % | 27/27 = 100 % |

All 15 differences are `abstain` tasks that the old scorer failed because its
phrase list did not contain the phrasings the models actually used — "does not
state", "does not specify", "does not provide". There are **zero** cases where
the old scorer passed and the new one failed, so the added strictness
(abstention guard, hedged-guess guard, token-level negation) introduced no
false negatives on this data.

**This mattered for the headline.** The old scorer would have ranked
nemotron-3 as the clearly most reliable model (100 % vs 67 % and 78 %) — a
ranking produced entirely by gaps in a phrase list, not by model behaviour.
All three models are in fact indistinguishable on this suite, and at n = 27
the 95 % interval is 88-100 %, which cannot separate them anyway.

Structured output found one genuine, reproducible failure: qwen-3.6 fails
`pso-v1-006-simple-routing` in all three repetitions by emitting `notify` as a
list where the schema requires a string. `consistency_rate` was 100 % for
every model, i.e. Ollama at `temperature = 0` is deterministic across
seed-varied repetitions.

### Full HumanEval, and what the token budget did to it (v0.6.0, 2026-08-01)

> **RETRACTED — the numbers in this section are invalid.** The harness was not
> executing the tests it reported on. `_build_program` appended the fixture's
> `tests` block, which only *defines* `check(candidate)`, and never called it,
> so the sandboxed module compiled, exited 0, and every syntactically valid
> answer counted as correct. The pass rates below therefore measure "does the
> output parse and import", not correctness. Fixed in the following commit
> (with a regression test that a deliberately wrong solution must fail); the
> section is kept for the record and superseded by the re-run below.

First run of the complete 164-problem set against the v1 lineup via Ollama
(`results/benchmarks/20260731T221321Z-9d630d4a`, 110 min total):

| Model | pass@1 @ 512 tok | pass@1 @ 1536 tok | genuine failures |
|---|---|---|---|
| qwen-3.6 | 98.8 % (CI 96-100) | **100 %** (CI 98-100) | 0 |
| gemma-4 | 94.5 % (CI 90-97) | **97.0 %** (CI 93-99) | 5 |
| nemotron-3 | 94.5 % (CI 90-97) | **96.3 %** (CI 92-98) | 6 |

**16 of the 20 failures were the token budget, not the models.** Every one
stopped at exactly 512 decode tokens, mid-function, and the sandbox reported
the resulting `SyntaxError` as `compile_error` — indistinguishable, in the
summary, from a model that writes broken code. Re-running only those tasks at
1536 tokens recovered 9 of 16 (qwen 2/2, gemma 4/8, nemotron 3/6); the rest
failed again well under the new budget, so those are real.

Consequences worth remembering:

- **The ranking changed shape.** At 512 tokens gemma-4 and nemotron-3 tie
  exactly; at 1536 they separate slightly, and qwen-3.6 goes from "nearly
  perfect" to unblemished. A budget that is too small penalises verbose models
  specifically, which is a property of their output style, not their coding.
- `configs/experiments/code-generation.yaml` now sets `max_tokens: 1536`. The
  1536-token column above comes from a targeted re-run of the truncated tasks
  only, so treat it as a corrected estimate — a clean full run at the new
  budget has not been done yet.
- HumanEval is saturated for this class of model. With every model at 96-100 %
  and intervals overlapping, this suite no longer separates them; the honest
  read is "all three solve essentially all of HumanEval". Ranking them on it
  would be reading noise. A harder set (or a contamination-controlled one) is
  needed for real discrimination.

### Full HumanEval, corrected harness (v0.6.0, 2026-08-01)

Re-run after two harness bugs were fixed: the sandbox never invoked
`check(candidate)` (so pass@1 measured "does it parse"), and prompt-defined
helper functions were not available to the tests. Bundle
`20260801T013617Z-74a9a46b`, 164 problems, 1 sample, `max_tokens = 512`:

| Model | pass@1 | 95 % CI | assertion | runtime | compile | wrong fn name |
|---|---|---|---|---|---|---|
| gemma-4 | **93.9 %** | 89-97 % | 4 | 0 | 6 | 1 |
| qwen-3.6 | **92.1 %** | 87-95 % | 11 | 0 | 2 | 0 |
| nemotron-3 | **74.4 %** | 67-80 % | 26 | 9 | 7 | 7 |

**The suite discriminates again.** Under the broken harness all three sat at
94-99 % and were indistinguishable; nemotron-3 in particular reported 94.5 %
and actually scores 74.4 %. Its interval (67-80 %) does not come close to the
other two, so that gap is real. gemma-4 and qwen-3.6 overlap heavily and
should be treated as tied.

Failure modes differ in kind, not just count. qwen-3.6 fails almost entirely
on assertions — code that runs and returns the wrong answer. nemotron-3 fails
across every category, including 7 tasks where it renamed the required
function and 9 runtime errors, i.e. a share of its gap is instruction-following
and output hygiene rather than algorithmic ability.

Caveats to carry forward:

- `max_tokens = 512` truncated a share of the failures (10 / 9 / 16 by model),
  so all three numbers are floors. `code-generation.yaml` now defaults to 1536;
  these results predate that.
- HumanEval is in the training data of every model here. High scores should be
  read as "not obviously broken at coding", not as a capability measurement.

### First full MBPP run, and a config default that undid a fix (2026-08-01)

426 sanitized MBPP problems against the same lineup, bundle
`20260801T035913Z-e9f504d5`, 1 sample, 4 h 26 min total:

| Model | MBPP pass@1 | 95 % CI | HumanEval pass@1 | 95 % CI |
|---|---|---|---|---|
| gemma-4 | **85.2 %** | 82-88 % | 93.9 % | 89-97 % |
| qwen-3.6 | **83.1 %** | 79-86 % | 92.1 % | 87-95 % |
| nemotron-3 | **71.6 %** | 67-76 % | 74.4 % | 67-80 % |

**MBPP is the less saturated set.** Every model loses 9-10 points against its
HumanEval score, except nemotron-3, which was already low. With n = 426 the
intervals are roughly half as wide, and nemotron-3 separates cleanly from the
other two (67-76 % vs 79-88 %); gemma-4 and qwen-3.6 still overlap and remain a
tie. The ranking is identical across two independent sets, so nemotron-3's
position is not an artefact of HumanEval.

**This run used the wrong token budget, and the reason is worth recording.**
`configs/experiments/code-generation.yaml` was raised to `max_tokens: 1536` the
day before, but `scripts/run_full_code_generation.py` defaulted to
`configs/experiments/ollama-baseline.yaml`, which still caps at 512 — so the
fix never reached the run that needed it. A config fix is only as good as the
default that selects the config. The runner default now points at
`code-generation.yaml`.

Truncation at 512 tokens accounted for a large share of the failures, and
re-generating exactly those tasks at 1536 gives a corrected estimate:

| Model | truncated failures | recovered at 1536 | corrected pass@1 | 95 % CI |
|---|---|---|---|---|
| gemma-4 | 32 of 63 | 10 | 87.6 % | 84-90 % |
| qwen-3.6 | 20 of 72 | 11 | 85.7 % | 82-89 % |
| nemotron-3 | 57 of 121 | 5 | 72.8 % | 68-77 % |

The recovery rates matter more than the corrected scores. qwen-3.6 recovers
11 of 20 (55 %): its truncated answers were mostly on their way to a correct
solution. nemotron-3 recovers 5 of 57 (9 %): its long answers were long *and*
wrong, so truncation was mostly a symptom, not the cause. The obvious reading
of "half of nemotron's failures are truncation, so its score is badly
understated" is therefore wrong — the gap survives the correction almost
intact.

Treat the corrected column as an upper-leaning estimate, not a measurement:
only previously failing tasks were re-generated, so tasks that passed at 512
had no chance to regress under a different sample. The clean full run at 1536
supersedes it.

### Clean MBPP run at the intended budget (2026-08-01)

Bundle `20260801T135412Z-54cefdaa`, 426 problems, `max_tokens = 1536`, 4 h
57 min:

| Model | pass@1 | 95 % CI | truncated | assertion | runtime | compile |
|---|---|---|---|---|---|---|
| gemma-4 | **87.6 %** | 84-90 % | 1 | 48 | 4 | 1 |
| qwen-3.6 | **85.7 %** | 82-89 % | 3 | 53 | 6 | 2 |
| nemotron-3 | **72.8 %** | 68-77 % | 4 | 90 | 21 | 5 |

Truncation is effectively gone (1 / 3 / 4, against 32 / 20 / 57 at 512), so
these numbers describe the models rather than the budget. nemotron-3 stays
clearly separated; gemma-4 and qwen-3.6 overlap and remain a tie, as they do
on HumanEval.

**The clean run reproduced the targeted re-check exactly** — 373 / 365 / 310
problems passed, the same counts the partial re-generation predicted. At
temperature 0 with a fixed seed that is the expected outcome, but it is worth
recording that the cheap method did not mislead here.

nemotron-3's failures differ in kind, not only in count: 90 assertion failures
against gemma-4's 48, i.e. code that runs and returns the wrong answer. Its
gap is not an output-format artefact.

### Portability check on an unseen model (2026-08-02)

`gpt-oss:120b` — a family, quantization (MXFP4) and size the harness had never
run, with no curated config — probed with
`scripts/check_model_portability.py`, five tasks per suite:

| | pass | empty | truncated | median tokens |
|---|---|---|---|---|
| code, reasoning off | 5/5 | 0 | 0 | 896 |
| code, reasoning on | 4/5 | 0 | **1** | 970 |
| grounding, off / on | 5/5 | 0 | 0 | 84 / 105 |
| structured, off / on | 5/5 | 0 | 0 | 164 / 164 |

**A model config and nothing else was enough.** No prompt, scorer or fixture
change; the config auto-detected from `/api/tags` ran all three suites clean.

Two things worth carrying forward:

- **The old 512-token budget would have scored this model as unable to code.**
  Its median code answer is 896 tokens with reasoning off. Nothing about that
  failure would have looked like a budget problem in the summary — it would
  have read as `compile_error`, exactly as it did for the v1 lineup before the
  truncation flag existed.
- **Reasoning mode costs budget, and 1536 is not much headroom.** One code
  answer in five hit the ceiling with reasoning on. That is below the
  threshold that flags a suite as unmeasurable, which is why the probe now
  reports any truncation at all as a caveat: it is not a hint that something
  may be wrong, it is a scored failure with a known cause, and one in five is
  the difference between 100 % and 80 %.

Five tasks per suite cannot rank a model, and this run does not try to. It
answers the prior question — whether the harness can measure it at all.

### Four models, and the bug the fourth one found (2026-08-02)

First run with a model outside the v1 lineup. Bundle
`20260802T051407Z-51ad314a`, MBPP, 426 problems, `max_tokens = 1536`,
reasoning off for every model, 7 h 53 min. Scores below are from the re-scored
bundle (`-rescored`); see the extraction bug beneath the table.

| Model | pass@1 | 95 % CI | truncated | assertion | runtime | compile |
|---|---|---|---|---|---|---|
| gemma-4 | **87.6 %** | 84-90 % | 1 | 48 | 4 | 1 |
| qwen-3.6 | **85.7 %** | 82-89 % | 3 | 53 | 6 | 2 |
| gpt-oss-120b | **78.9 %** | 75-82 % | **48** | 27 | 18 | 45 |
| nemotron-3 | **73.0 %** | 69-77 % | 4 | 91 | 19 | 5 |

gemma-4 and qwen-3.6 reproduced their previous run exactly (373 and 365
problems), which is the expected outcome at temperature 0 and a useful check
that nothing else moved.

**The new model found a scoring bug on its first run.** `extract_code` took
the first fenced block in the output. gpt-oss-120b explains before it answers
and its explanations contain fences — a bare regex on `mbpp/7`, a pseudocode
"Algorithm" block on `mbpp/14` — so the harness compiled those and recorded
the `SyntaxError` as the model's failure. It scored 54.9 % with 134 of 426
tasks as `compile_error`, a failure profile no competent model produces:
weak code *runs* and returns wrong answers, it does not fail to parse.

Blocks are now chosen by content (the one defining the entry point, else the
last one defining anything). Re-scoring the stored answers:

| Model | before | after | changed |
|---|---|---|---|
| gpt-oss-120b | 234 | **336** | +102, −0 |
| nemotron-3 | 310 | 311 | +1, −0 |
| qwen-3.6 | 365 | 365 | 0 |
| gemma-4 | 373 | 373 | 0 |

Zero regressions, and the asymmetry is the finding: the bug was invisible for
as long as only the v1 three were measured, because all three answer with a
single fence. **Three models cannot validate a harness that claims to measure
any model.** This is the fifth scoring bug in this suite, and the first found
by widening the lineup rather than by reading the code.

**gpt-oss-120b's 78.9 % is a floor, and more so than the others.** 48 of its
90 remaining failures are answers cut off at 1536 tokens; the corresponding
counts are 1, 3 and 4 for the other three. Its median code answer is 775
tokens against gemma-4's 318 and qwen-3.6's 114. Every model was given the same
budget, which is the only way the comparison is readable — but a budget that
suits three models is itself a measurement choice, and it is not neutral
between them. What this table supports is "at 1536 tokens, on MBPP,
gpt-oss-120b places third". What it does not support is a claim about its
coding ability in general.

### Long-context retrieval (v0.4.0 – v0.4.3, fast profile)

Fast profile grid: lengths = 4 096 / 32 768 / 131 072 tokens,
depths = 0 % / 50 % / 100 %, 2 needles per cell = 18 cells per model.

**Key pattern: depth dominates length.** Every tested model—qwen-3.6, gemma-4,
nemotron-3, and gpt-oss:120b-cloud—passes at depth 100 % (needle at the *end* of
the context) and fails at depths 0 % and 50 % essentially uniformly, regardless of
context length. This is the classic "lost in the middle" failure, not a
memory or bandwidth limitation of the test machine.

Selected results from the v0.4.3 run (20260602):

| Model | passes / 18 | depth-0 % | depth-50 % | depth-100 % | VRAM @ 131k |
|---|---|---|---|---|---|
| qwen-3.6 | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 28 880 MB |
| gemma-4 | 6 / 18 = 33 % | 0/6 | 0/6 | 4/6 | 34 430 MB |
| nemotron-3 | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 31 173 MB |
| gpt-oss:120b-cloud | 6 / 18 = 33 % | 0/6 | 0/6 | 6/6 | 67 537 MB |

**Prefill throughput scales with memory bandwidth.** Qwen-3.6 achieves
~1 450 tok/s at 4k context, degrading to ~1 120 tok/s at 131k. Nemotron-3 is
similar (~1 940 → 1 520). Gemma-4 is notably slower (~655 → 360) due to its
architecture. The 120B cloud model processes ~1 210 → 780 tok/s.

**Version-over-version prompt improvements.** The v0.4.0 run (pre-v0.4.2) showed
depth-0 and depth-50 scoring 0 on all models because models treated the needle as
"not in context". The v0.4.2 anti-refusal prompt and numeric-format normalisation
resolved the refusal issue; the pattern above is now genuine retrieval failure,
not prompt phrasing.

**What this means.** The machine's 128 GB unified memory is *not*
the bottleneck—every model loads to 131k without OOM, and prefill throughput stays
high. The bottleneck is model architecture: these models are not trained to retrieve
facts from mid-context positions. This is the honest finding. The quant sweep
(v0.5.0) should show whether smaller quantisations shift the failure point.

**Next step.** Run the full grid (4×4×8) for at least qwen-3.6 and nemotron-3 to
produce statistically significant per-depth columns. The fast profile confirms the
pattern; the full profile will quantify it precisely enough to publish.
