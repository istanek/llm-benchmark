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
