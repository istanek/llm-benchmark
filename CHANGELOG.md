# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The report recommends on cost when quality ties.** The overall ranking
  sorts by score, so it named gemma-4 the default pick over qwen-3.6 — a model
  it cannot separate statistically and which costs 15x less energy per solved
  task. The recommendation now takes every model whose interval overlaps the
  leader's and prefers the cheapest measured, stating both numbers. Quality
  still comes first: a model outside the leader's interval is never
  recommended on price.
- **The verdict no longer claims axes it did not measure.** It read "strongest
  combined result across reliability and speed" in a run with no speed suite,
  where the 0.20 speed weight had silently dropped out of the score.
- **Energy per solved task** (`llm_benchmark.energy`, wired into the code
  suite) — GPU power sampled while each model answers, integrated
  trapezoidally over sample timestamps, reported as J/solved and tasks/Wh
  with the host's idle draw alongside. First result: qwen-3.6 and gemma-4
  score identically on a 30-task subset and gemma-4 costs 15x the energy
  (1 869 J vs 125 J per solved task, 1.9 vs 28.8 tasks/Wh). The gap is wall
  clock rather than draw — power spans 1.5x across the lineup, time spans
  10x — so energy is throughput in other units, and the fix for an expensive
  model is a faster model.
- **Long-context results** (`RESULTS.md`) — every model retrieves a needle at
  every depth up to 131k tokens, so retrieval joins grounding and structured
  output as a suite that cannot rank this lineup. Prefill throughput at 131k
  does separate them, by 7x: nemotron-3 2 498 tok/s against gemma-4's 344,
  which is fifty seconds of prefill against six minutes. gemma-4 leads on code
  and trails badly here, so the recommendation depends on input length.
- **Preflight before every run** (`llm_benchmark.preflight`, plus
  `llm-bench preflight`) — fixtures load, external corpora are on disk, model
  tags are pulled, the backend answers, and no model asks for a capability its
  backend lacks. Runs automatically at the top of every bundle and in the
  standalone code runner, and can be invoked on its own before an unattended
  sweep. Written because the 2026-08-02 sweep measured four models for eight
  hours and then died on its last suite over two Project Gutenberg texts that
  had never been downloaded — checkable in milliseconds. Every problem is
  reported at once, each with the command that fixes it: stopping at the first
  would mean finding the second after the next eight-hour attempt. It reports
  and refuses; it never repairs, because a preflight that quietly fetches a
  corpus or drops a suite decides what the run measures.
- **First contamination result** (`RESULTS.md`) — no model in the lineup shows
  a confirmed signal on MBPP. gemma-4, qwen-3.6 and nemotron-3 drop by under
  2.5 points on the reworded set (p = 0.71 / 0.86 / 0.21). gpt-oss-120b drops
  5.2 points at p = 0.013, but rewording also pushed its truncation from 52 to
  73 answers; on the 336 tasks neither run truncated, the drop is p = 0.099.
  The apparent signal is largely a verbose model meeting a fixed budget.
- **Grounding v2 turned out to be saturated as well** — 14/14 for three models
  and 13/14 for nemotron-3 once the scorer defects were fixed. The spread
  reported earlier in the day was mostly those defects, not the models.
- **Non-English answers are `unscorable`, not failures.** Every phrase list in
  the grounding scorer is English, so a correct refusal in Czech, German,
  Spanish, Russian or Japanese scored as a hallucination — a verdict about the
  scorer's vocabulary wearing the costume of a result. Such rows now leave the
  denominator and are reported separately in both the summary and the report.
  Detection is function-word density plus script detection, stdlib only, and
  runs only on answers that already failed, so it cannot reclassify a pass.
  Short answers ("2019", "Dvorak") are never guessed at. Follows the rule the
  portability probe already applies: a bad score is a result, an unscorable
  output is not a measurement.
- **`scripts/paired_compare.py`** — McNemar's exact test on per-task outcomes,
  plus a paired bootstrap CI and the task-level flip lists. Every model answers
  the same fixture, and comparing marginal Wilson intervals throws that away:
  on the four-model MBPP bundle it turns the gpt-oss-120b / nemotron-3 "tie"
  into a separation (61 vs 36 flips, p = 0.014), while confirming the gemma-4 /
  qwen-3.6 tie as a measured result rather than an absence of evidence. The
  flip lists are printed before the p-value on purpose: flips concentrated in
  one task family, or a re-run of the same model flipping both ways, is news
  about the scorer rather than the model. Contributed as a review suggestion;
  its truncation probe only descended into dicts, so `--exclude-truncated`
  silently dropped zero tasks on the code suite, where truncation decides the
  ranking — fixed and pinned by tests.
- **Grounding v2 wired in as a provisional suite** —
  `hallucination_grounding_v2`, 14 near-miss tasks where a plausible wrong
  answer sits inside the context. v1 scores 100 % for every model and cannot
  say which one invents less; v2 gives qwen-3.6 14/14, nemotron-3 13/14 and
  gpt-oss-120b 11/14. Adds a fourth scored behaviour, `report_conflict`: when
  the context contradicts itself, every conflicting value must appear and the
  disagreement must be named. Quoting both and picking a side anyway is the
  failure it exists to catch.
- **`--suite` and `--model` on `llm-bench benchmark`** — run exactly what is
  named, bypassing the keyword router. An unattended multi-hour run should not
  depend on whether the request happened to contain the right noun.
- **`configs/experiments/full-sweep.yaml`** — every axis in one bundle,
  ordered cheapest-first so a tail that drags costs only the tail.
- **Mutated MBPP as a contamination probe** —
  `data/code/code_generation_mbpp_mutated_v1.json`, the same 426 problems with
  renamed entry points, paraphrased prose and recomputed example arguments,
  generated by `scripts/build_mbpp_mutants.py` and validated by executing every
  canonical solution through the real sandbox. Wired through the suite
  resolver, orchestration dispatch, TUI registry, `--benchmark mbpp-mutated`
  and the rescore tool. Read as a delta against MBPP: both sets are the same
  problems, so a model that drops further than the others is showing recall
  rather than ability. The fixture canary now covers it, including two
  generator bugs it caught — a dropped trailing newline that broke any model
  echoing the prompt, and renaming that mangled the problem statement.
- **Verbosity reported as its own axis** (`llm_benchmark.verbosity`, plus a
  section in both reports) — median / p90 / p99 answer length, answer density
  (the share of the output that was the answer rather than prose about it),
  decode tokens per solved task, and a pass@1 floor–ceiling bracket from
  truncation. A quantile that lands on the token budget is marked censored
  rather than reported as a measurement. Written because gpt-oss-120b's score
  was mostly a statement about the budget: its bracket spans 11 points where
  the other three span less than one.
- **First four-model comparison** (`METHODOLOGY.md`) — gemma-4 87.6 %,
  qwen-3.6 85.7 %, gpt-oss-120b 78.9 %, nemotron-3 73.0 % on MBPP at a shared
  1536-token budget with reasoning off. gemma-4 and qwen-3.6 reproduced their
  earlier run exactly. gpt-oss-120b's number is a floor in a way the others'
  are not: 48 of its 90 failures are truncation, against 1-4 for the rest.
- **`scripts/rescore_bundle.py`** — apply the current scorer to a finished
  bundle's stored answers and write a new bundle, instead of spending hours
  re-asking the models. Scoring is offline and deterministic; neither the
  prompt nor the sampling config references the scorer. The original bundle is
  left intact, and provenance records both halves (`git_commit` = the scorer,
  `generations_from` = the run that produced the answers) so a re-scored bundle
  cannot read as a fresh measurement.
- **The report says what was asked of each model.** A "Run conditions" block in
  both the markdown and HTML reports lists per-model `reasoning`, effective
  `max_tokens`, quantization and tag, plus the harness commit and whether the
  tree was clean. Once the budget is per-model a pass rate stops being
  self-describing: a model handed four times the budget just looks better in
  the table. Runs from different commits are flagged as not readable side by
  side, and bundles with no provenance say so instead of appearing complete.
- **Truncated answers are counted in the report.** A `truncated` column per
  model, and a note that the pass rates are floors when any answer hit the
  budget. Truncation is a scored failure with a known cause and was previously
  invisible outside the raw results.

### Fixed

- **The grounding scorer failed four correct answers.** Found by running the
  new fixture against real models before the sweep it was built for; each
  would have reported gpt-oss-120b as hallucinating more than the others.
  Value matching now folds diacritics (`Dvořák` vs `Dvorak`), thousands
  separators between digits (`2,600` vs `2600`) and markdown emphasis;
  `ABSTAIN_PHRASES` accepts assertions of absence ("did not acquire any
  company in 2019") and the "the context only gives X" family that a near-miss
  context invites; and a fixture can veto a fabricated value after an
  abstention with `rejected_values`.
- **The reliability suites hard-capped answers at 256 tokens.**
  gpt-oss-120b named a conflict correctly and was cut off before quoting the
  second value. Cap is now 512 — measured need is a p90 of 181 tokens for the
  most verbose model in the lineup — and the same for every model, because a
  per-model budget here would make grounding scores incomparable.
- **`benchmark` ignored the suites its experiment declared.** The plan came
  from the request text, so a config listing five suites ran a hardcoded trio
  when the request matched no keyword; the first launch of the full sweep
  started `openclaw_speed`. The experiment's own suites are now the fallback,
  with keyword routing still winning when the request names something.
- **One v2 fixture context never named the entity its question asked about**,
  so a model that pointed that out was scored as failing to report a conflict.
- **`extract_code` took the first fenced block, not the one with the answer.**
  A model that explains before it answers — gpt-oss-120b opens with a bare
  regex or a pseudocode "Algorithm" block — had its explanation compiled and
  the resulting `SyntaxError` scored as its own failure: 134 of 426 MBPP tasks,
  putting it at 54.9 % instead of 78.9 %. Blocks are now chosen by content (the
  one defining the entry point, else the last one defining anything). Re-scoring
  the stored answers moved gpt-oss-120b +102 tasks and nemotron-3 +1, with zero
  regressions — the bug was invisible for as long as only single-fence models
  were measured.
- **Provenance was stamped on request rather than by default**, and the first
  run started after it shipped came out unstamped: the standalone
  code-generation runner builds its own manifest and had not been updated.
  `build_manifest` now defaults to the checkout it was imported from.


## [0.7.0] - 2026-08-02

### Added

- **MBPP sanitized as a second code benchmark** —
  `data/code/code_generation_mbpp_v1.json`, 426 problems from
  google-research/google-research (CC BY 4.0). HumanEval is saturated for this
  class of model (94-99 % before the harness fixes, 74-94 % after) and is in
  everyone's training data, so a second, larger set gives an independent read
  and tighter intervals. Tasks carry `metadata.benchmark = "mbpp_sanitized"`,
  matching the entries `reference_scores.yaml` has always had.
  - Wired through the suite-name resolver, bundle dispatch, TUI registry, the
    `run` CLI, and `scripts/run_full_code_generation.py --benchmark mbpp`.
  - MBPP prompts are prose, so each prompt embeds the first assert to pin the
    signature the tests expect. That is standard for MBPP and is part of the
    measurement: the model is asked to match a given signature, not invent one.
  - All 426 canonical solutions execute through this repo's sandbox as a
    conversion check. `mbpp/56` is excluded — its canonical solution recurses
    past Python's limit — and the exclusion is stated in the fixture notes.

- **`scripts/bench`** — the unattended-run launcher, previously an untracked
  file in `~`. Rewritten to track the run by pid: `pgrep -f
  run_full_code_generation` matched any process whose command line merely
  mentioned that string, so a shell waiting for the run to end matched itself,
  waited forever, and `./bench status` reported a run that had finished hours
  earlier as still going.
- **Launchers so the CLI runs from anywhere** — `scripts/llm-bench-launcher`
  plus `scripts/install-launchers.sh`, which put `llm-bench`, `llm_benchmark`,
  `spark_benchmark` and the other declared names on `$PATH`. No install step:
  this system's Python is externally managed (PEP 668), and the last
  `pip install -e .` left three scripts in `~/.local/bin` importing
  `spark_benchmark` — a module that stopped existing at the rename, so every
  one of them had been failing with `ModuleNotFoundError`. The launcher also
  reports a missing checkout in a sentence, since the repo lives under `/tmp`
  and does not survive a reboot. `LLM_BENCH_REPO` points it elsewhere.
  `spark_benchmark` (underscore) is now a declared entry point too.
- **`llm-bench compare <bundle> --baseline <bundle>`** — the workflow the
  harness exists for: run a new model, find out where it lands against models
  already measured, without re-running them. Verdicts are `better` / `worse` /
  `tie` by non-overlapping 95 % Wilson intervals. It **refuses** across a
  provenance mismatch — different harness commit, dirty tree, missing stamp,
  different fixture version, or a shared model run with different options —
  because a verdict across that boundary is indistinguishable from a real one.
  `--force` prints it under a recorded objection.
- **Runs stamp their provenance** — `RunManifest.provenance` records the git
  commit, whether the working tree was clean, and the options each model
  actually ran with (`reasoning`, effective `max_tokens`, quantization, tag).
  `suite_version` was never enough: both bugs that invalidated this project's
  code results were code changes under an unchanged fixture version. Bundles
  written before this field have none and cannot serve as a baseline without
  `--force`.
- **Per-model reasoning mode and output budget** — `ModelConfig` gains
  `reasoning` and `max_output_tokens`. Both were previously constants tuned to
  the v1 lineup: the Ollama payload hardcoded `think: false`, so a reasoning
  model was measured with its reasoning pass off, and one experiment-wide
  `max_tokens` truncated any model more verbose than those three. A backend
  that cannot honour `reasoning` now refuses the model
  (`BackendCapabilities.supports_reasoning`) instead of running it in the other
  mode without saying so. The override is resolved in the adapter, so the
  recorded request shows the budget actually sent.
- **`scripts/check_model_portability.py`** — takes any Ollama tag (curated
  config optional) and runs a few tasks from the code, grounding and
  structured-output suites, reporting the four harness-side failure modes —
  empty answers, truncation, unparseable code, abstentions the scorer's phrase
  list misses — separately from the pass rate. Exits non-zero when the model is
  not measurable as configured, and prints the config lines that would fix it.
- **Draft of a harder grounding fixture, for review** —
  `data/reliability/hallucination_grounding_v2_draft.json`, 14 tasks. Not
  resolvable by suite name, not referenced by any config, and not run by
  anything: v1 is at 100 % for all three models, so it cannot say which model
  invents less, but replacing the fixture that judges the models is a decision
  to take deliberately. Every task keeps a plausible wrong answer within reach
  (adjacent attribute, same value on a neighbouring entity, same entity in
  another role), and each carries a `review_note` saying what it targets. Two
  conflicting-source tasks are marked `needs-scorer-support`: scoring them
  needs a `report_conflict` behaviour that does not exist yet.
- **First full MBPP results** (`METHODOLOGY.md`) — gemma-4 85.2 %, qwen-3.6
  83.1 %, nemotron-3 71.6 % over 426 problems. MBPP is 9-10 points below
  HumanEval for the top two, and n = 426 halves the intervals, so nemotron-3
  separates from the other two beyond doubt while gemma-4 and qwen-3.6 stay
  tied. Includes the truncation re-check and why it does *not* rescue
  nemotron-3.

### Fixed

- **The full-run script defaulted to the wrong experiment config.**
  `scripts/run_full_code_generation.py` used
  `configs/experiments/ollama-baseline.yaml` (`max_tokens: 512`) rather than
  `configs/experiments/code-generation.yaml`, where the budget had just been
  raised to 1536 — so the 4.5 h MBPP run silently ran at the old budget and
  truncated 32-57 answers per model mid-function. Default now points at
  `code-generation.yaml`.

- **`extract_code` dropped code defined above the entry point.** It sliced from
  `def <entry_point>`, keeping only preceding imports and decorators, so a
  model that writes a helper function or a module-level constant above the
  target function had them cut away and the tests died with `NameError` —
  scored as the model's failure. Extraction now starts at the first top-level
  construct (`def` / `class` / `import` / `@` / assignment). Found while
  validating the MBPP conversion, where it accounted for 9 of 12 canonical
  solutions failing; 426/427 pass after the fix. HumanEval results are
  unchanged, since those answers are usually fenced and take a different path.

## [0.6.1] - 2026-08-01

### Added

- **`tests/test_fixture_canary.py` — the harness must prove it can tell right
  from wrong.** Both bugs fixed in this release had the same shape: the
  pipeline reported plausible pass rates while not testing what it claimed.
  The canary runs known-good code (canonical solutions) and known-bad code (a
  stub with the right signature returning `None`) through the real pipeline and
  asserts both verdicts, over a deterministic sample of the fixture plus the
  tasks with prompt-defined helpers. Validated by reintroducing each bug: the
  missing `check()` call trips `test_stubbed_solutions_fail`, the missing
  helper injection trips `test_canonical_solutions_pass`. Runs in ~1.3 s, so it
  guards every CI run.

### Fixed

- **Prompt-defined helper functions were missing from the sandbox.** A few
  HumanEval prompts define a helper above the stub (`humaneval/32` uses `poly`,
  `humaneval/38` uses `encode_cyclic`) and the canonical tests call it. The
  official harness runs `prompt + completion`, so the helper is always there;
  we ran the extracted answer, which usually holds only the target function, so
  the test died with `NameError` and the model was blamed for a harness
  artefact. `_build_program` now re-injects any prompt-defined function the
  answer does not redefine. This turned 2 of qwen-3.6's failures and 4 of
  gemma-4's into passes on identical model output.
- **The code sandbox never ran the tests it reported on.** `_build_program`
  appended the fixture's `tests` block — which only *defines*
  `check(candidate)` — and never emitted the `check(<entry_point>)` call. The
  sandboxed module therefore compiled, exited 0 and was recorded as a pass, so
  `pass@1` measured "does the output parse and import", not correctness. A
  deliberately wrong solution (`return False` for `has_close_elements`) scored
  as correct. The invocation is now appended and three regression tests pin it,
  including the wrong-solution case that would have caught this originally.
  **All previously published `code_generation` numbers are invalid**, including
  the full-HumanEval table added earlier today; METHODOLOGY.md marks that
  section as retracted.

### Changed

- **`code-generation.yaml` raises `max_tokens` from 512 to 1536.** The first
  full HumanEval run showed 16 of 20 failures across three models were
  generations cut off at exactly 512 decode tokens, not wrong answers; the
  longest problems need ~1200. See METHODOLOGY.md for the corrected pass@1
  table and what the small budget did to the ranking.

### Fixed

- **Ollama truncation was invisible.** `OllamaAdapter` mapped every completed
  response to `finish_reason = "stop"`, discarding the `done_reason` Ollama
  actually sends (`"length"` when the reply hit `num_predict`). The truncation
  flags added in 0.6.0 therefore could never fire on the Ollama backend — the
  grounding run reporting zero truncated rows was uninformative, not
  reassuring. The adapter now passes `done_reason` through.
- **Truncated code samples were reported as `compile_error`.** A generation cut
  off mid-function fails the sandbox with a `SyntaxError`, which
  `_classify_failure` filed as broken code rather than an exhausted token
  budget. `SampleOutcome.truncated` is now recorded per sample, written to
  `results.jsonl`, and collected as `truncated_task_ids` per benchmark, so
  "the model cannot solve this" is distinguishable from "max_tokens was too
  low". Found on the first full HumanEval run: both of qwen-3.6's two failures
  in 164 problems are truncations at exactly 512 decode tokens, not wrong
  answers.
- `TRUNCATION_FINISH_REASONS` / `is_truncated` moved to `models.py` so the code
  suite and the reliability suites share one definition; `"incomplete"` joins
  the set.

### Added

- **`scripts/run_full_code_generation.py`** — unattended runner for the full
  164-problem HumanEval set across several models. Each model writes into its
  own run directory inside one bundle, so an interrupted session loses at most
  the model in flight; re-running against the same `--output-dir` skips models
  that already produced a `summary.json`. A failing model is logged and the
  run continues rather than losing the night. Ends by rendering the bundle
  report and printing measured pass@1 per model — it deliberately does not
  write those into `reference_scores.yaml`, since checking a measurement
  against itself would validate nothing.

### Changed

- **NVIDIA telemetry moved into the `telemetry` package.** NVML and
  `nvidia-smi` polling lived inline in `sustained_throughput.TelemetrySampler`,
  leaving NVIDIA as the one vendor that did not go through the collector
  protocol the portable-core work introduced. It is now
  `telemetry/nvidia.py::NvidiaTelemetryCollector`, selected by
  `build_telemetry_collector` ahead of Apple and AMD (it is the only collector
  reporting throttle reasons) and falling through to the stub when neither NVML
  nor `nvidia-smi` is present. The sampler keeps only threading, cadence and
  the timestamped sample; GPU clock and throttle reasons now travel through the
  same snapshot dict as every other field. Verified on an NVIDIA GB10: power,
  temperature and clock are captured, and `memory.used` stays absent rather
  than being charted as 0 MB.

### Added

- **`code_generation` fixture holds the full HumanEval set** (0.1.0 → 0.2.0):
  164 problems from openai/human-eval (MIT), up from the 5-problem starter
  subset — the extension the fixture's own description called for. All 164
  canonical solutions were executed through this repo's sandbox as a
  conversion check (164/164 pass). At n = 5 a pass@1 carried a 95 % interval
  spanning most of the scale; at n = 164 it is a publishable number.
- **`--task-limit` / `task_limit=`** for the code suite, since a full run is
  now 164 problems per model. A truncated run records `task_limit`,
  `tasks_available` and `partial_run: true` in its summary, so a short smoke
  check can never be mistaken for a full-set score. Verified against qwen-3.6
  (`--task-limit 8` → 8/8 pass@1, `partial_run: true`).

## [0.6.0] - 2026-07-31

### Validated

- **The hardened grounding scorer was checked against a real run** rather than
  stub backends only (`results/benchmarks/20260731T205220Z-93efc960`:
  qwen-3.6 / gemma-4 / nemotron-3 via Ollama, 9 tasks x 3 repetitions).
  Re-scoring those outputs with the old implementation gives qwen-3.6 78 %,
  gemma-4 67 %, nemotron-3 100 % — against 100 % for all three under the new
  one. All 15 differences are abstentions the old phrase list missed
  ("does not state" / "specify" / "provide"), and there are zero regressions
  in the other direction. See METHODOLOGY.md for the full table.

### Fixed

- **`openai-compatible` decode throughput was understated.** The adapter
  published the full request round trip as `decode_time_s`, and reporting
  derives tok/s as `decode_tokens / decode_time_s` — so queueing, prefill and
  network all counted as decode time. The adapter now **streams by default**
  and measures time-to-first-token, making `decode_time_s` a real decode
  window and `ttft_ms` a real measurement instead of an unavailable metric.
  `options.stream: false` restores the single-shot path, which now declares
  `decode_time_s: round_trip_only_includes_prefill_and_network` in its
  capabilities rather than passing the round trip off as decode time.
  Capabilities are selected per mode. Note: tok/s from the Laguna smoke run
  (`20260731T170649Z`, 30.58 tok/s) was measured with the old accounting and
  is a lower bound.
- The same file was rewritten from single-line statements into the layout used
  by the rest of the package.

### Changed

- **Project renamed from `spark-benchmark` to `llm-benchmark`.** The harness is
  no longer DGX Spark-specific, so the naming no longer claims it is:
  - Python package `spark_benchmark` → `llm_benchmark` (`src/llm_benchmark/`).
  - Console scripts: `llm-bench` (Typer CLI) and `llm-benchmark` /
    `llm_benchmark` (TUI shell). The old `spark-bench` / `spark-benchmark`
    entry points are kept as deprecated aliases and will be removed in 1.0.
    **Re-run `pip install -e .` after pulling** — the old package directory is
    gone, so a stale editable install will fail to import.
  - CLI banner art now reads `LLM`; the panel title is "LLM Benchmark". The
    `SPARK_BENCH_NO_BANNER` env var still works alongside the new
    `LLM_BENCH_NO_BANNER`.
  - HTML report hero breadcrumb reads "LLM BENCHMARK" and the default tagline
    no longer names any vendor.
- **Platform config generalised.** `configs/platforms/spark.yaml` →
  `configs/platforms/local.yaml` (`name: local`, `display_name: Local
  machine`, auto-detected architecture/OS). Concrete CPU/GPU/memory facts
  already come from `hardware.discover_hardware()` per run.
  **`--platform spark` is now `--platform local`**; `shell.DEFAULT_PLATFORM`
  changed accordingly.
- **Experiment configs lost their `spark-` prefix**: `spark-ollama-baseline.yaml`
  → `ollama-baseline.yaml`, and likewise for `llamacpp-baseline`,
  `trt-reliability`, `code-generation`, `sustained`, `quant-sweep`,
  `laguna-smoke`. Internal experiment names dropped the prefix too
  (`spark-ollama-v1-baseline` → `ollama-v1-baseline`), so aggregates keyed on
  the old names will not match older runs.
- **Fixture content de-branded, with version bumps where the change is
  behavioural:**
  - `openclaw_speed_v1` 0.1.0 → 0.2.0 — the probe prompt asked the model to
    explain what DGX Spark is; it now asks about local LLM benchmarking.
    Timing results are comparable across the change, generated text is not.
  - `practical_structured_output_v1` 0.1.0 → 0.2.0 — calendar task title
    "Spark Demo Rehearsal" → "Q3 Demo Rehearsal" (context and expected JSON
    changed together).
  - `quantization_sweep_v1` — description/notes only, no version bump.
- Documentation (`README.txt`, `docs/`, `METHODOLOGY.md`, Cursor rules)
  reworded from "Spark-only" to "single-machine". Hardware is still named in
  the three places where it changes how a measurement must be read (unified
  memory hiding `nvidia-smi memory.used`, and the provenance of the v0.4.x
  long-context findings).

### Fixed

- **Grounding scorer no longer passes outputs that contradict the expected
  behaviour.** Three concrete defects in `reliability.score_hallucination_task`:
  - `NEGATION_PHRASES` were matched as substrings, so `"no"` matched inside
    *nothing* / *north* / *know* and `"not"` inside *cannot* / *note*. Any
    output of moderate length therefore counted as a negation, and
    `correct_user` collapsed into a plain token-overlap check — "Nothing to
    note: 1998." scored as a correct rejection of a false premise. Phrase
    matching is now whole-token (`tokenize` + `contains_phrase`).
  - `answer_from_context` passed when *any single* reference token appeared
    anywhere in the output, so a refusal that quoted the value ("The context
    does not mention 1998") scored as a correct answer of 1998. Scoring now
    uses the fixture's explicit `expected_values` contract (falling back to
    literal reference match or ≥ 60 % content-token coverage) and fails
    outright when the output abstains.
  - `abstain` matched a fixed English phrase list, letting a hedged
    fabrication through on `"unknown"` ("…is unknown to me, but it was likely
    Dr. Novak") while rejecting common phrasings such as "provides no
    information". The list gained the missing phrasings and a hedged-guess
    guard (`HEDGED_GUESS_PHRASES`) now vetoes abstention-plus-fabrication.
- **`repetitions` and `warmup_runs` are read.** Both fields existed on
  `ExperimentSpec` (defaults 3 and 1) and were set by every experiment YAML,
  but no runner ever read them: every suite ran each task exactly once with no
  warmup. `run_benchmark_bundle` and `cli run` now pass them to the grounding,
  structured-output, speed and sustained suites. Each repetition offsets the
  seed so repeats are not trivially identical.
- **Grounding token budget raised from 64 to 256** (`GROUNDING_MAX_TOKENS`).
  The old cap truncated models that emit a reasoning preamble before their
  answer and scored the truncation as a hallucination. Truncated rows are now
  flagged (`evaluation.truncated`, reason suffix `+truncated_output`) in both
  reliability suites so a budget that is still too small is visible in the
  results.
- **Timing probes no longer report a synthetic 100 % pass rate.**
  `openclaw_speed` marks its summary `scoring: "performance_probe"`; markdown,
  CLI and HTML reports render `n/a` in the pass column for probe suites (with
  name-based detection so pre-0.5.3 runs are handled too), and the HTML speed
  card shows timed-generation counts instead of three identical pass bars.
  The overall ranking was already unaffected — `openclaw_speed` is not in
  `_QUALITY_SUITE_KEYS`.

### Added

- **`llm_benchmark.stats` — Wilson score intervals.** Pass rates are now
  reported with their sample size and a 95 % interval in the markdown report,
  the CLI benchmark summary, per-suite `summary.md`, and the HTML per-model
  table (new `n` / `95% CI` / `Consistency` columns). At the v1 suite sizes the
  intervals are wide, which is the point: a one-task difference should not read
  as a result. `aggregate_runs` exposes `ci_low` / `ci_high` / `ci_margin` per
  model.
- **Repetition consistency reporting.** `build_summary` returns `tasks`,
  `repetitions`, `unstable_task_ids`, `consistency_rate` and `truncated_rows`
  per model; `failed_task_ids` is deduplicated across repetitions. This is the
  "consistency across repeated runs" axis METHODOLOGY.md asks for.
- **Explicit fixture value contract.** `data/reliability/hallucination_grounding_v1.json`
  (0.2.0 → 0.3.0) declares `expected_values` on every `answer_from_context` and
  `correct_user` task, plus `rejected_values` (the false premise) on the latter.

### Tests

- 22 new tests in `tests/test_scoring_hardening.py` pinning each fixed failure
  mode: token-vs-substring matching, sycophantic agreement, abstention that
  quotes the value, hedged guesses, truncation flagging, seed-varied
  repetitions, the 256-token grounding budget, consistency accounting,
  performance-probe tagging, and Wilson-interval behaviour at small n and at
  the 0 % / 100 % extremes.

## [0.5.2] - 2026-06-07

### Fixed

- **Local and cloud models can now be benchmarked in the same run.** The
  Ollama adapter previously resolved the generation endpoint once at startup
  from `$OLLAMA_HOST`, so whenever `OLLAMA_HOST=https://ollama.com` was set
  (needed for cloud runs), every model — including locally-pulled ones like
  `qwen3.6:35b` — was sent to ollama.com and returned 404. Generation routing
  is now per-model: local models always go to `localhost:11434` (or the
  configured backend endpoint), cloud models (`source=ollama-cloud` or a
  `-cloud` tag) go to `https://ollama.com`. The API key is sent only for
  cloud-routed requests, never to a local endpoint.
- **TUI Cloud menu now sets `OLLAMA_HOST` alongside `OLLAMA_API_KEY`.** The
  `do_cloud` prompt previously set only the API key, which fixed detection
  (dual-probe uses the key to query ollama.com) but left generation routing
  pointing at localhost. After entering a key the menu now also sets
  `OLLAMA_HOST=https://ollama.com`; entering `-` clears both.
- **Model discovery always probes `localhost:11434`, ignoring `OLLAMA_HOST`.**
  When `OLLAMA_HOST=https://ollama.com` was set the local probe was
  misdirected to ollama.com without auth, so local models disappeared from
  the TUI picker. Detection now always probes the local daemon directly and
  independently probes Ollama Cloud when `OLLAMA_API_KEY` is present.

### Tests

- 6 new tests in `tests/test_model_registry.py`: `model_is_cloud` detection,
  and `OllamaAdapter` routing for local/cloud/private-remote-host cases.

## [0.5.1] - 2026-06-06

### Added

- **`llm_benchmark.quant_sweep` — post-processor and HTML tradeoff table.**
  - `aggregate_quant_sweep(aggregate, model_configs, fixture)` groups
    `aggregate_runs()` output by `base_model × quantization` using the
    `base_model` field already present on `ModelConfig`. Suite-name version
    suffixes (e.g. `hallucination_grounding_v1`) are stripped automatically.
  - `check_quant_regressions(sweep, fixture)` returns warning strings when
    any non-reference variant drops more than 5 pp below the reference
    threshold for a suite. Only fires when `enforce: true` on the base-model
    spec (currently `false` for all v1 entries until baselines are measured
    on hardware).
  - `load_quant_sweep_fixture(path)` loads and validates
    `data/quant/quantization_sweep_v1.json` via Pydantic.
  - `_render_quant_sweep_card` and `_render_quant_sweep_section` added to
    `reporting_html`. The card renders a per-base-model tradeoff table
    (Variant · Quant · Hallucination · Struct. output · Code pass@1 · TTFT ·
    tok/s · VRAM). Quality cells are colour-coded relative to the reference
    variant (green ≥ ref, amber within 5 pp, red > 5 pp below). Reference
    row is sorted first; remaining variants follow fixture order.
  - `render_canonical_report_html` now accepts `aggregate["quant_sweep"]` or
    an explicit `quant_sweep=` kwarg — the section is omitted when absent,
    so existing reports are unchanged.
  - `enrich_with_quant_sweep(aggregate, model_configs, repo_root, fixture_path)`
    added to `quant_sweep`. Detects quant-sweep runs automatically (any model
    with `base_model` set), loads the fixture, calls `aggregate_quant_sweep`,
    and injects the result into the aggregate dict in place. No-ops silently
    when no model carries a `base_model` or the fixture file is missing.
  - `cli.py` `benchmark` and `wizard` commands and `shell.py` TUI `do_run`
    now call `enrich_with_quant_sweep` after `aggregate_runs`, so the quant
    tradeoff table appears automatically in the HTML report whenever quant
    variants are benchmarked together.
- **21 new tests in `tests/test_quant_sweep.py`** covering fixture loading,
  aggregation grouping, suite-version stripping, missing-suite → null,
  regression enforcement (enforce=False silent, enforce=True fires, within-5pp
  silent, null-threshold skipped), and HTML smoke tests for the card and
  section renderers.

### Fixed

- `tests/test_long_context.py`: `test_existing_model_yaml_still_loads_without_base_model`
  renamed to `test_model_config_loads_without_base_model` and switched to a
  synthetic inline YAML so the test doesn't break when the real `qwen-3.6.yaml`
  gains a `base_model` field (as it now has).

## [0.5.0] - 2026-06-05

### Added

- **BYOT `mode: scored` — deterministic scorers (v0.3.0 milestone, shipped
  in v0.5.0 alongside the quant sweep infrastructure).**
  `CustomSuiteTask` now accepts an optional `scoring:` block. Five scorers:
  - `exact_match` — normalised case-insensitive string equality.
  - `substring_match` — all items in `must_contain` must appear in the output.
  - `regex_match` — a Python `re` pattern must match somewhere in the output.
  - `json_fields_match` — output must parse as JSON and contain all
    `expected_fields` keys with the expected values. Markdown fences are
    stripped automatically.
  - `multiple_choice` — a letter/word expected answer must appear as a whole
    word in the output.
  A suite-level `scoring:` block provides the default for tasks that don't
  specify their own. Tasks without any scorer are still run but produce
  `passed: null`. `validate_custom_suite` now warns when `mode: scored` is
  used but no scorer is configured for a task.
  `load_custom_suite` no longer rejects `mode: scored`.
- **`--dry-run` flag on `llm-bench run-custom`.** Executes one task against
  one model and stops without writing any files. The JSON output carries
  `"dry_run": true` and a single row. Useful for sanity-checking backend
  connectivity and suite config before a long sweep.
- **Per-task `timeout_s` enforcement.** `run_custom_suite_quick` now wraps
  each backend call in a `_task_timeout` context manager. On Unix platforms
  (where `SIGALRM` is available) a timeout triggers `TimeoutError`, which is
  recorded as an error row and the run continues. On Windows / non-main
  threads the timeout is a no-op (field is still recorded for forensics).
- **Quantization sweep infrastructure (v0.5.0 preview).**
  - `docs/quantization-sweep-spec.md` — implementation-ready spec.
  - `data/quant/quantization_sweep_v1.json` — fixture with reference
    thresholds for the v1 lineup (all `enforce: false` until baselines run).
  - `configs/experiments/quant-sweep.yaml` — experiment covering Q8_0
    and Q4_K_M variants of all three base models.
  - Six new model config YAMLs: `qwen-3.6-q8`, `qwen-3.6-q4`, `gemma-4-q8`,
    `gemma-4-q4`, `nemotron-3-q8`, `nemotron-3-q4`.
  - `base_model` field added to the three existing curated model YAMLs
    (`qwen-3.6`, `gemma-4`, `nemotron-3`).
  The post-processor and HTML tradeoff table shipped in v0.5.1.
  Pull the quant variants (`ollama pull qwen3.6:35b-q8_0` etc.) and run the
  experiment; canonical suites produce per-model results that
  `aggregate_quant_sweep` groups into the tradeoff card automatically.
- **Long-context empirical findings in `METHODOLOGY.md`.** Documents the
  "lost in the middle" pattern across all four tested models (depth drives
  retrieval more than context length), prefill throughput numbers, and the
  v0.4.2 prompt-improvement rationale.

### Changed

- `run_custom_suite_quick` progress callback now includes the scorer method
  in the task line (`[exact_match]`) and prints `PASS` / `FAIL` with a
  reason snippet after each scored task.
- `render_custom_summary_markdown` renames "Per-model telemetry" to
  "Per-model summary" and adds Pass / Pass rate / Scored columns in
  `mode: scored`.
- `build_custom_summary` per-model buckets now always include `passes`,
  `scored`, and `pass_rate` fields (all zero/null in `mode: quick`).
- `validate_custom_suite` no longer emits an error for `mode: scored`
  (it now validates instead of rejecting).

### Tests

- 25 new tests in `tests/test_custom_suites.py` covering:
  - `ScoringConfig` schema validation (all 5 methods, missing required fields).
  - `score_response` for all scorers including edge cases (case-sensitivity,
    word-boundary for multiple_choice, JSON with markdown fences, invalid
    regex, missing JSON keys).
  - Runner integration: mode: scored pass/fail, suite-level default scorer,
    unscored task → null verdict, dry_run no files written.
  - `validate_custom_suite` warning for unscored tasks in scored mode.

## [0.4.5] - 2026-06-02

### Added

- **`run --model` (repeatable).** The `run` command can now target specific
  models by name/tag instead of always running the full resolved lineup. It
  accepts a curated experiment name, the raw Ollama tag, its slugified form,
  or an explicit Ollama Cloud `-cloud` tag — the latter is synthesized on the
  fly, so cloud models work without an experiment YAML entry or `/api/tags`
  listing. Example:
  `run --experiment … --platform local --run-suite hallucination_grounding --model gpt-oss:120b-cloud`.

### Fixed

- **Docs:** the Ollama Cloud examples in `README.txt`, `docs/README.md`, and
  the v0.4.4 release notes used a non-existent `run --suite … --model …`
  invocation. Corrected to real commands (`quick --models …` and
  `run … --run-suite … --model …`) and listed the valid `--run-suite` values.

## [0.4.4] - 2026-06-02

### Added

- **Ollama Cloud support.** Benchmark hosted models (e.g.
  `gpt-oss:120b-cloud`, `deepseek-v3.1:671b-cloud`) with no config changes —
  set `OLLAMA_HOST=https://ollama.com` and `OLLAMA_API_KEY=…` and pick a
  cloud model by tag. The Ollama adapter now resolves its base URL from
  `$OLLAMA_HOST` (falling back to the configured endpoint, then localhost)
  and sends an `Authorization: Bearer` header from `$OLLAMA_API_KEY` on every
  request (generate / unload / `/api/ps` / `/api/tags`). New helpers
  `resolve_ollama_base`, `ollama_auth_headers`, `is_cloud_endpoint`. Model
  detection is auth-aware, and `find_config_by_name_or_tag` synthesizes a
  config for an explicit `-cloud` tag even when `/api/tags` doesn't list it
  (`synthesize_cloud_model_config`).
- **Security:** the API key is read from the environment only and is never
  copied into `BackendConfig.options`, manifests, or report payloads.

### Notes

- Cloud runs report speed (tokens/s, TTFT) and pass rates, but **no local
  GPU telemetry** — memory/power/temperature are unavailable remotely, so
  those charts stay empty. Cloud calls are billed and traverse the network,
  so long suites are slower.

## [0.4.3] - 2026-06-02

### Added

- **Long-context HTML report: pass-rate by needle type.** A multi-model
  run showed needle *category* drives retrieval as much as position does —
  e.g. alphanumeric codes (~17% across all models) are far harder than
  dates (~67%), because models garble or loop on codes even when they're
  clearly trying. `build_long_context_summary` now emits a per-model
  `categories` breakdown ({category, passes, n, pass_rate}), it's forwarded
  through `aggregate_runs`, and the report renders a model × needle-type
  pass-rate heatmap. `_svg_heatmap` gained a `left_pad` argument so long
  model names fit the row gutter.
- **`scripts/probe_refusal.py`** — a standalone diagnostic (stdlib +
  Ollama HTTP, no install needed) that varies only the prompt wording
  (baseline / instruction-at-end / forceful) at a fixed short context. It
  established that the depth-0 retrieval collapse is **not** a wording or
  refusal artifact (all three variants score identically), so the
  anti-refusal prompt is not the lever — needle type and position are.

## [0.4.2] - 2026-06-01

### Changed

- **Long-context scoring now ignores thousands separators.** A correct
  numeric answer was being failed purely on formatting — e.g. a model
  answering `1840` when the planted fact said `1,840`. `score_niah` now
  strips digit-group separators (comma, space, NBSP, narrow NBSP,
  apostrophe) that sit *between two digits*, so `1840` == `1 840` ==
  `1,840`. Non-numeric punctuation (such as the comma in
  `November 14, 2023`) is untouched, and genuinely different numbers still
  fail.
- **Needle-in-a-haystack prompt now states the fact is present.** A
  multi-model run showed every model scoring 0 % whenever the needle was
  *not* at the very end of the context — they treated the planted fact as
  out-of-place in the public-domain filler and refused ("not answerable").
  The prompt now uses standard NIAH framing ("a specific fact … has been
  inserted … the answer is present, do not reply that it is missing"), so
  the suite measures retrieval rather than refusal. Note: this changes
  prompt semantics, so long-context pass rates from 0.4.0/0.4.1 are not
  directly comparable to 0.4.2+.

## [0.4.1] - 2026-06-01

### Added

- **Fast / full profiles for the long-context suite.** The full
  needle-in-a-haystack grid is 4×4×8 = 128 cells per model and, dominated
  by long-context prefill (the 131k row alone is ~half the time), runs
  roughly an hour per model. There's now a **fast preview profile**
  (3 lengths `4096 / 32768 / 131072` × 3 depths `0/50/100` × 2 needles =
  18 cells, ~10 min/model) that still spans the full range up to 131k. It
  shows up in the TUI suite picker as a separate
  `long_context_retrieval_fast` entry and works on the CLI via
  `run --suite long_context_retrieval_fast`. Profiles live under a new
  `profiles` block in the fixture and are validated like the default grid;
  `LongContextFixture` gained `profiles`, plus `resolve_profile_matrix` /
  `profile_for_suite_name` helpers and a `matrix` override on the runner.

### Changed

- **Long-context suites are now opt-in in the TUI "Run" picker.** Because
  they're slow and the two profiles are mutually redundant, neither
  long-context entry is preselected by default — the five quick canonical
  suites stay checked and you tick the one long-context profile you want.
  The `Suites`/`Info` screens report the fast entry's actual (smaller)
  grid.

## [0.4.0] - 2026-06-01

### Added

- **Long-context retrieval in the interactive TUI + HTML reports (layer 3).**
  `long_context_retrieval` now shows up in the `llm-bench shell` suite
  picker and `Suites`/`Info` screens (grid-aware task counts:
  lengths × depths × needles/cell). A preflight checks the git-ignored
  Project Gutenberg corpora are present and cleanly skips the suite with a
  "run scripts/fetch_haystacks.sh" hint instead of crashing mid-run. The
  HTML report gains a dedicated long-context card: a per-model
  length×depth **pass-rate heatmap** (red→amber→green, N/A tiles for
  lengths a model can't load) via a new dependency-free `_svg_heatmap`
  helper, a **prefill-throughput-vs-length** line chart, and a
  **resident-memory-vs-length** chart (shown only when Ollama `/api/ps`
  yielded memory), plus a first-failure-length KPI strip. The aggregator
  forwards the per-cell grid through `aggregate_runs`.
- **`long_context_retrieval` runner (layer 2 of the v0.4.0 suite).**
  Single-needle NIAH execution across the fixture grid. For each model it
  iterates the `length × depth × needles_per_cell` matrix and writes one
  of three states per cell: `pass`/`fail` (ran and substring-scored),
  `skipped_unsupported` (length exceeds the model's claimed context), or
  `error` (backend raised — e.g. OOM — captured, never fatal). Built on
  the probe findings: each cell prompt carries a deterministic-but-unique
  nonce to defeat Ollama's prefill cache, `options.num_ctx` is set
  explicitly per request, the backend request timeout is bumped to
  ≥ 600 s for long prefills, and the reported context length is the
  backend's actual `prompt_eval_count` (never the char-based estimate).
  Memory is sourced from Ollama `/api/ps` (new
  `OllamaAdapter.memory_snapshot()`) since `nvidia-smi` reports N/A on the
  Spark's unified memory. Summary aggregates per-cell pass rates, average
  prefill tok/s, peak VRAM, and a first-failure-length per model, plus a
  per-model length×depth markdown heatmap table. Wired into both the
  orchestration bundle and the `run --suite long_context_retrieval` path;
  added `SamplingConfig.num_ctx`. Reporting/HTML heatmaps land in layer 3.
- **Design spec for `long_context_retrieval` (v0.4.0 target).** New
  `docs/long-context-spec.md` is the implementation-ready plan for the
  single-needle NIAH suite, written against the real v0.3.0 codebase.
  Locks in: Project Gutenberg / Apache-2.0 public-domain haystacks,
  Part A (substring scoring) only — no LLM judge, per-model
  tokenization with honest reported lengths, a 4×4×8 grid (128
  tasks/model, 8 samples/cell) for statistically meaningful heatmaps,
  inline-SVG heatmaps via a new `_svg_heatmap` helper (no matplotlib /
  no new runtime dependency), deterministic needle/haystack selection,
  and three-state cells (pass / N/A-unsupported / OOM). Supersedes the
  aspirational Suite 1 sketch in `docs/extensions-spec.md`. Also records
  the agreed release ladder: v0.4.0 long-context → v0.5.0
  quantization sweep → v0.6.0 concurrent serving.

### Fixed

- **TUI: Esc now cleanly exits a selection back to the menu.** Pressing
  Esc/`q` in the model or suite multiselect (Run / Custom / Quick) was
  conflated with confirming an empty selection, so it printed a stray
  "(no models selected)" notice that looked like a dead-end sub-screen.
  Esc-cancel (`None`) is now distinguished from an empty confirm and
  returns silently to the main menu.

## [0.3.0] - 2026-06-01

### Added

- **Marketing-grade HTML reports — second pass.** The HTML reports
  picked up a complete visual overhaul on top of the standalone-file
  foundation introduced earlier in this release. Same single-file
  invariants (no JavaScript, no CDN, no external assets), but the
  visual quality is now "would happily attach this to a board deck"
  rather than "minimum viable HTML":
  - **Hero banner** at the top of every report — radial-gradient
    purple/indigo background (cyan-leaning for custom runs so the
    two flavours are visually distinct), display-size H1, subtitle
    pulled from the request prompt, and a glassmorphism "Recommended
    pick" winner card with the model name and a one-line justification
    ("perfect grounding reliability; TTFT 120 ms; 42.0 tok/s").
  - **Stat-tile strip** under the hero — five tiles for canonical
    bundles (models tested, suites run, total tasks, overall pass
    rate colour-graded, top model score) and four for custom runs
    (completed pairs, errored pairs, fastest decode model, lowest
    TTFT model).
  - **Verdict card** with a soft gradient background and indigo accent
    border, replacing the bare-bones verdict paragraph.
  - **Color-coded pass-rate cells.** Every percentage cell in the
    canonical report tables now carries a CSS ``--cell-pct`` custom
    property and a ``data-band`` (good / warn / bad / na) attribute,
    rendering a proportional fill behind the value (≥95 % green,
    80–95 % amber, <80 % red).
  - **Sticky table headers** so column labels stay visible while
    scrolling long rankings.
  - **Print stylesheet** (``@media print``) — gradients flatten to
    flat colours, shadows vanish, ``<details>`` collapses cleanly,
    every ``break-inside`` is set to avoid cutting tables / cards.
- **Per-suite dashboard cards with suite-specific charts.** Each of
  the five canonical suites now renders into a dashboard card with
  a 3-up grid of charts tailored to *that* suite, plus the shared
  per-model results table:
  - **``openclaw_speed``** — pass rate (good-bg gradient) + TTFT bar
    chart with **inverted colour** (lower = greener = better) +
    decode throughput (tok/s) bars in green.
  - **``hallucination_grounding``** and
    **``practical_structured_output``** — pass rate + TTFT
    (inverted) + a wide **per-task pass-fail strip** (green / red /
    grey squares per task) loaded lazily from the suite's
    ``results.jsonl``.
  - **``code_generation``** — aggregate pass@1 + **per-benchmark
    stacked bars** (HumanEval / MBPP / …) + a wide **sandbox-status
    breakdown** (passed / failed / timeout / oom / compile_error /
    runtime_error) loaded from per-row sandbox status fields.
  - **``sustained_throughput``** — initial vs sustained
    **dual-bar** per model + per-model **throttle-ratio gauges**
    (semicircle SVG arcs, colour-graded) + per-model **peak-temp
    thermometers** + a wide **tps-over-time line chart** with an
    optional GPU-temperature overlay (dashed secondary axis) loaded
    from ``telemetry-<model>.jsonl``.
- **New SVG primitives** in ``reporting_html``: ``_svg_line_chart``
  (multi-series with optional secondary axis, adaptive grid lines,
  inline legend), ``_svg_gauge`` (180° semicircle, colour-graded,
  optional invert), ``_svg_dual_bars`` (paired thin bars with shared
  scale), ``_svg_stacked_bars`` (segmented bar with hint counts),
  ``_svg_thermometer`` (vertical bar + bulb), ``_pass_fail_strip_html``
  (per-model task-by-task dot strip). All inline-SVG, all
  ``viewBox``-based so they scale with the container.
- **Lazy data loaders** for the renderer:
  ``_load_results_rows(run_dir)`` reads ``results.jsonl`` (skips
  malformed lines, returns empty on missing file),
  ``_load_telemetry_samples(run_dir, model, max_points=240)`` reads
  ``telemetry-<model>.jsonl`` and uniformly downsamples (a
  30-minute soak with ~18 000 points compresses to a 240-point
  curve under a few KB).
- **Custom (BYOT) / quick run polish.** ``summary.html`` now ships
  with the same hero / stat-tile chrome (cyan-tinted gradient so it's
  visually distinct from a canonical bundle), each ``<details>``
  task block opens with a 2-up mini-chart row showing **TTFT
  comparison** (lower is better, inverted colour) and **output
  length** (decode tokens) per model, and the task summary header
  carries a small **error strip** of dots so the user can scan a
  long suite for failures without expanding every block.
- **Plumbing in ``aggregate_runs``.** Per-model entries now forward
  ``windows`` (sustained-throughput per-window throughput series),
  ``benchmarks`` (code-generation per-benchmark breakdown), and
  ``run_dir`` (so the HTML renderer can lazy-load
  ``results.jsonl`` / ``telemetry-*.jsonl`` without re-walking the
  filesystem). Markdown / CLI summaries are unchanged.
- **27 new tests** in ``tests/test_reporting_html.py`` covering
  every new SVG helper (line chart with secondary axis, gauge with
  invert, dual bars, stacked bars normalising per-row, thermometer,
  pass-fail strip), color helpers (``_gradient_color_for_ratio``,
  ``_band_for_pass_rate``, ``_cell_pct_html``), lazy loaders
  (results.jsonl + telemetry downsampling + missing-file
  fallbacks), suite-specific dispatch (all 5 suites + unknown-suite
  fallback), color-coded ranking cells, and a true end-to-end
  reliability render that builds a ``results.jsonl`` on disk and
  asserts the per-task strip survives the round-trip. Total HTML
  test count: 38 (was 11).

- **Polished standalone HTML reports.** New
  ``llm_benchmark.reporting_html`` module renders both flavours of
  run output as a single self-contained HTML page — no JavaScript,
  no CDN, no external assets. Open the file from a USB stick, attach
  it to an email, paste it into a wiki: it just works.
  - Canonical bundles now emit ``report.html`` next to ``report.md``
    with overall ranking, per-suite tables, narrative commentary,
    verdict / recommendation, and inline SVG bar charts for overall
    score and per-suite pass rates.
  - Custom (BYOT) and ``quick`` runs now emit ``summary.html`` next
    to ``summary.md`` / ``summary.json`` with a per-model telemetry
    table, mean-decode-tps bar chart, and one collapsible
    ``<details>`` block per task showing every model's reply
    side-by-side. Errored cells are highlighted in red.
  - ``write_report`` learned a new ``"both"`` format that writes the
    ``.md`` and ``.html`` siblings in one call. Existing
    ``"markdown"`` / ``"html"`` paths continue to work unchanged.
  - All renderers HTML-escape user content (prompt text, model
    output, error messages) so YAML suites containing ``<script>``
    or ``<img onerror=...>`` payloads can't escape the ``<pre>`` /
    ``<code>`` containers.
- **CLI / TUI surface for HTML.** ``llm-bench benchmark``,
  ``wizard``, ``aggregate``, ``run-custom``, and ``quick`` all log
  the HTML path next to the existing markdown / JSON paths.
  ``aggregate``'s JSON output gained ``"aggregate_html"`` and the
  custom commands gained ``"summary_html"``. The TUI ``Run`` /
  ``Custom`` / ``Quick`` flows print the HTML path in their final
  log block.
- **``tests/test_reporting_html.py``** — 11 plain-Python tests
  covering document well-formedness (doctype, no script tags,
  embedded ``<style>``), canonical-renderer ranking / verdict /
  per-suite blocks, custom-renderer telemetry / per-task details,
  HTML-escaping of user-supplied prompts and outputs, SVG bar-chart
  edge cases (empty input, value formatting, clamping), and a
  ``write_report(..., "both")`` integration assertion that the
  ``.md`` and ``.html`` siblings land next to each other.

- **Quick (ad-hoc one-shot prompts).** New
  ``llm_benchmark.quick`` module surfaces the lightest BYOT
  workflow yet — type one prompt, fan it out to every model you
  picked, get the same ``summary.md`` ``run-custom`` produces. No
  YAML required up front.
- **CLI command ``llm-bench quick "your prompt here"``.** Builds
  a one-task ``CustomSuiteDefinition`` in memory
  (``task_id="ad-hoc"``) and feeds it to the existing
  ``run_custom_suite_quick`` runner — single runner, single
  summary format, single results layout. Flags: ``--models``,
  ``--allow-auto-detected`` (default ON), ``--name`` (overrides the
  ``quick-<slug>`` default), ``--save`` / ``--save-path`` /
  ``--overwrite`` to persist the prompt as a reusable suite YAML,
  and ``--output-dir``.
- **TUI menu entry ``Quick``.** Sits between ``Custom`` and
  ``Models``. Walks the user through model multi-select → drops
  out of curses to read a single-line prompt on the regular TTY →
  runs ``run_custom_suite_quick`` with progress streaming into the
  log → asks ``Save this prompt as a reusable custom suite?
  [y/N]`` afterwards. If saved, the run's ``manifest.json`` is
  patched in place so ``suite_path`` points at the saved YAML and
  ``discover_custom_suites`` surfaces it next time.
- **Saved-quick layout.** ``examples/custom-tests/quick-saved/`` is
  the default save root. The directory is **git-ignored** (added
  to ``.gitignore``) so personal one-shots stay out of source
  control while still being findable by the existing TUI discovery
  helper.
- **Manifest provenance fields.** Quick runs carry
  ``source: "cli-quick" | "shell-quick"`` and
  ``ad_hoc_prompt: true`` so reports can tell quick runs apart
  from canonical custom-suite runs (which use ``"cli"`` /
  ``"shell"``).
- **``tests/test_quick.py``** — 12 plain-Python tests covering
  ``build_quick_suite`` (one task, ad-hoc id, default-name slug,
  empty-prompt rejection, sampling pass-through, punctuation-only
  fallback), ``save_quick_suite_as_yaml`` (round-trip through
  ``load_custom_suite``, refuses to clobber by default,
  ``overwrite=True`` replaces, empty optional fields trimmed), and
  an end-to-end run against a fake backend asserting one row per
  ``(model, ad-hoc)`` pair.

### Changed

- ``shell.MENU_ITEMS`` gained ``("quick", "Quick")``; dispatch
  routes Enter on it to ``TUIApp.do_quick``.
- ``CONTRIBUTING.md`` is unchanged; ``README.txt``,
  ``docs/README.md``, ``docs/architecture.md``,
  ``docs/custom-tests-spec.md``, and
  ``.cursor/rules/project-overview.mdc`` were extended to cover
  the new entry point.
- ``reporting.render_html_report`` is now a thin delegate to
  ``reporting_html.render_canonical_report_html``. The previous
  unstyled stub (a bare ``<html><body>`` with one un-themed table
  per suite) is gone — anything that called it now produces the
  full styled report instead, which is a deliberate behaviour
  change with no downstream API change.
- The benchmark / wizard / shell-run flows now write the report
  bundle as ``"both"`` (``report.md`` *and* ``report.html``) by
  default. ``aggregate`` likewise writes both.

### Fixed

- **Sustained-throughput dual bars overflowed their panel.** The
  "Initial vs sustained throughput" SVGs in
  ``_svg_dual_bars`` sat one level deeper inside an extra flex
  wrapper, so the ``.bar-row svg { flex: 1 1 auto }`` rule didn't
  reach them. With no explicit ``width``, browsers fell back to
  the inline-SVG default (300 px) and the bars spilled into the
  neighbouring "Throttle ratio" card. The track wrapper now uses
  ``flex: 1 1 0; min-width: 0; overflow: hidden`` and the inner
  SVGs carry ``width: 100%; display: block``. Added matching
  CSS safety nets — ``.bar-row svg`` now also sets
  ``min-width: 0``, ``.dual-bars-track svg`` enforces full-width,
  and a global ``svg.lines { width: 100%; height: auto }`` rule
  keeps the throughput line chart honest in narrow containers.

## [0.2.1] - 2026-05-28

Polish release on top of 0.2.0. Surfaces the BYOT subsystem in the
curses TUI (so users no longer have to type ``run-custom`` flags),
moves the project home from GitLab to GitHub (history and tags carry
over with identical commit hashes), and fixes a long-standing
"ESC needs two presses to leave a submenu" bug.

### Fixed

- **ESC double-press in the curses TUI.** ``ncurses`` defaults to
  ``ESCDELAY=1000``, so a bare ESC sat in the read buffer for a full
  second while the library waited to see if it was the start of an
  escape sequence (arrow keys, F-keys). Users learned to hit ESC
  twice. ``shell.TUIApp.run`` now calls ``curses.set_escdelay(25)``
  right after ``curses.curs_set(0)`` (the value vim and htop use),
  with a graceful fallback for environments where the symbol is
  missing. Single ESC now leaves singleselect / multiselect overlays
  immediately.

### Changed

- **Project home moved from GitLab to GitHub
  (`https://github.com/istanek/llm-benchmark`).** All Git history
  and both release tags (`v0.1.0`, `v0.2.0`) carry over unchanged
  (identical commit hashes). Knock-on edits in this commit:
  - ``CHANGELOG.md`` compare/tag link references repointed from
    ``gitlab.com/.../-/compare`` and ``-/tags`` to
    ``github.com/.../compare`` and ``releases/tag``.
  - ``docs/README.md`` swaps the GitLab pipeline badge for a GitHub
    Actions CI badge and rewords the "landing page" note.
  - ``README.txt`` "Help, support, bugs" now points at GitHub Issues
    and a pull request workflow; the install snippet uses the new
    GitHub URL.
  - ``CONTRIBUTING.md`` switches "Merge Request" / "MR" / "GitLab
    issues" wording to "Pull Request" / "PR" / "GitHub Issues" and
    points the cloning snippet at GitHub.
  - ``.gitlab-ci.yml`` was removed and replaced with
    ``.github/workflows/ci.yml`` running the same two stages
    (YAML/JSON fixture lint + ``pytest tests/``) on every push and
    pull request to ``main``.
  - ``scripts/release.sh`` was rewritten against the GitHub Release
    API (``POST /repos/<owner>/<repo>/releases``,
    ``Authorization: Bearer …``, ``Accept: application/vnd.github+json``).
    It now reads ``GITHUB_TOKEN`` first, then falls back to
    ``gh auth token`` and finally to ``~/.git-credentials`` for
    ``github.com``. The CHANGELOG-extraction logic and tag/push
    flow are unchanged.

### Added

- **Custom (BYOT) menu item in the curses TUI (`llm-bench shell`).**
  A new ``Custom`` entry sits next to ``Run`` and walks the user
  through the same flow as ``llm-bench run-custom`` on the CLI:
  it discovers suite YAMLs (shipped templates under
  ``examples/custom-tests/`` plus prior runs under
  ``results/custom/<slug>/<run-id>/``), loads + validates the
  selected suite, asks for models in a multiselect that respects
  the suite's ``models:`` list when present, and streams progress
  into the log as the run executes. The run bundle is written to
  ``results/custom/<slug>/<run-id>/`` with ``manifest.json`` tagged
  ``source: shell`` so reporting can tell TUI runs apart from CLI
  runs. ``--allow-auto-detected`` is implicitly on for the TUI
  entry, matching ``llm-bench run-custom``.
- **`shell.discover_custom_suites(repo_root)` helper** — pure
  function that returns ``CustomSuiteCandidate`` items, dedupes
  recent runs by absolute ``suite_path`` (newest ``run-id`` wins),
  and silently skips manifests pointing at deleted suite files.
  Covered by three new tests in ``tests/test_shell.py``.

## [0.2.0] - 2026-05-22

Second public release. Adds the Bring-Your-Own-Test (BYOT) subsystem
in Mode A (pass-through, no scoring), unifies model auto-detection
across every CLI surface behind one shared registry, and switches the
GitLab project page to the plain-language `README.txt`.

### Changed

- **Project landing page is now `README.txt` (plain language).** The
  markdown version moved to `docs/README.md` so the GitLab project page
  renders the human-friendly plain-text overview by default. PyPI
  metadata (`pyproject.toml::readme`) was repointed to the new
  location and serves the same markdown content. All cross-references
  in the `docs/` tree were updated to mention both paths; root
  `README.md` was deleted.

### Added

- **Bring-Your-Own-Test (BYOT) subsystem — Mode A.** New
  `llm_benchmark.custom_suites` module with a YAML / JSON suite
  format (`CustomSuiteDefinition`), a Pydantic-validated loader,
  resume-friendly runner that records errors per `(model, task)` pair
  without aborting, side-by-side Markdown summary, and a
  ``slugify_suite_name`` helper for run-bundle naming.
- **CLI commands `llm-bench run-custom` and `llm-bench validate-custom`.**
  `run-custom` defaults to ``--allow-auto-detected`` ON (custom suites
  exist precisely for non-curated workloads) and writes its bundles to
  ``results/custom/<slug>/<run-id>/`` with a manifest tagged
  ``kind: custom`` so reporting can keep these visually distinct from
  canonical suites. ``validate-custom`` exits non-zero on any error
  issue (duplicate task IDs, empty prompts, ``mode: scored`` not yet
  implemented, unknown model references).
- **Example custom suite template** at ``examples/custom-tests/quick/``
  with a working ``suite.yaml`` (Czech idiom translation, JSON
  extraction, Python code review) and a ``README.md`` explaining how to
  copy, edit, and run it.
- **Spec doc** ``docs/custom-tests-spec.md`` covering the v0.2.0 cut
  (``mode: quick`` only) plus the explicit roadmap for v0.3.0
  (deterministic scorers + ``dry-run``), v0.4.0 (sandboxed custom
  Python scorers + per-task timeout enforcement), v0.5.0 (local-only
  LLM-as-judge), and v0.6.0+ (sharing).
- **`tests/test_custom_suites.py`** — 16 plain-Python tests covering
  schema validation (duplicate IDs, empty prompts, empty tasks),
  YAML / JSON loaders, soft validation (long prompts, bad sampling,
  unknown model refs), end-to-end runner including resume + error
  recording, summary aggregation, Markdown rendering, and the
  ``slugify_suite_name`` helper.
- **Shared model registry** (`llm_benchmark.model_registry`) extracted
  from `shell.py`. One classification path is now used by the curses TUI,
  the wizard, the console REPL, the natural-language `benchmark` command,
  and the plain `run` command.
- **`--allow-auto-detected` flag** on `run`, `console`, `benchmark`, and
  `wizard`. When set, every chat-capable Ollama tag is offered alongside
  the curated experiment lineup, with auto-synthesized `ModelConfig`s
  carrying `notes=["auto-detected from Ollama (no YAML config)"]`. Off by
  default to preserve reproducibility for `run` and the NL routers.
- **`console --model` accepts Ollama tags directly** (`--model phi4:14b`)
  via `find_config_by_name_or_tag`, in addition to slugified
  (`phi4-14b`) and curated experiment names.
- **`tests/test_model_registry.py`** — coverage for `slugify_tag`,
  `synthesize_model_config`, `classify_detected`, the new
  `resolve_runnable_models` resolver (default + auto-detect + collision),
  and `find_config_by_name_or_tag` resolution order.

### Changed

- Curses TUI no longer owns its own classifier; it delegates to
  `model_registry.classify_detected`. The `classify_models(ctx, detected)`
  shape is preserved for backwards compatibility.
- `cli.detect_ollama_model_tags` is now a thin wrapper over the shared
  `detect_ollama_models`. The duplicate URL/JSON parsing logic is gone.

## [0.1.0] - 2026-05-20

Initial public release of the llm-benchmark scaffold.

### Added

- **Core harness**
  - YAML-driven experiment definitions validated through Pydantic v2
    (`ExperimentSpec`, `PlatformConfig`, `BackendConfig`, `ModelConfig`,
    `SamplingConfig`).
  - Typer CLI (`llm-bench`, `llm-benchmark`) with `run`, `console`,
    `benchmark`, `wizard`, `aggregate`, `report`, `dashboard`, `shell`
    subcommands.
  - Curses TUI shell with model / suite multiselect, chat mode, log
    follower, and live progress callbacks.
  - Run bundle layout: `results/benchmarks/<run-id>/<suite>/` containing
    `manifest.json`, `results.jsonl`, `summary.json`, `summary.md`.
- **Backend adapters**
  - Ollama HTTP adapter (production path for v1).
  - llama.cpp subprocess adapter (`llama-cli`).
  - Stub adapter used as fallback for `trt-llm` / `vllm`.
- **Suite runners**
  - `openclaw_speed` — TTFT and decode probe on short OpenClaw-like
    prompts (no quality scoring).
  - `hallucination_grounding` — grounded answers vs. abstention vs. false
    premise correction; heuristic scorer with abstention-phrase / negation
    / token-overlap rules.
  - `practical_structured_output` — exact-match JSON evaluation with
    fenced-block extraction and trailing-text rejection.
  - `code_generation` — HumanEval starter subset with sandboxed execution
    (`subprocess + resource.setrlimit + timeout`), `pass@k` unbiased
    estimator, and reference-score validation against
    `data/code/reference_scores.yaml`.
  - `sustained_throughput` — 5-minute decode soak per model with NVML or
    `nvidia-smi` telemetry, per-window aggregation, throttle ratio,
    energy per token.
- **Reporting**
  - Aggregator (`aggregate_runs`) joining manifests, summaries, and JSONL
    rows by suite × model.
  - Markdown / HTML / CLI summary renderers with overall ranking,
    per-suite commentary, and verdict paragraph.
- **Natural-language orchestration**
  - `parse_benchmark_request` — keyword + alias router that understands
    Czech and English (`rychlost`/`speed`, `spolehliv`/`reliab`,
    `kod`/`code`, `dlouhodob`/`sustained`, `openclaw`).
- **Fixtures (v1 starter sets)**
  - `data/reliability/hallucination_grounding_v1.json` — 9 tasks across
    `answer_from_context`, `abstain`, `correct_user`.
  - `data/practical/practical_structured_output_v1.json` — 6 JSON
    exact-match scenarios.
  - `data/performance/openclaw_speed_v1.json` — 3 short prompts for
    latency / throughput probing.
  - `data/performance/sustained_throughput_v1.json` — 3 long-form
    prompts cycled during decode soak.
  - `data/code/code_generation_v1.json` — 5 canonical HumanEval problems
    plus `data/code/reference_scores.yaml` template.
- **Configs**
  - `configs/experiments/`: `spark-ollama-baseline`,
    `spark-llamacpp-baseline`, `spark-code-generation`, `spark-sustained`,
    `spark-trt-reliability`.
  - `configs/models/`: `qwen-3.6`, `gemma-4`, `nemotron-3` plus a
    tombstone for the retired `nemotron-3-super`.
  - `configs/backends/`: `ollama`, `llamacpp`, `trt-llm`.
  - `configs/platforms/local.yaml`.
- **Tests** — plain-python, runnable via `pytest tests/` or `python3
  tests/test_<name>.py` (each file has a `_run_all()` fallback).
- **Docs** — `README.md`, `METHODOLOGY.md`, `docs/architecture.md`,
  `docs/extensions-spec.md`, `CONTRIBUTING.md`.
- **Cursor rules** — `.cursor/rules/{project-overview,python-conventions,
  fixtures-and-configs}.mdc`.
- **GitLab CI** — `.gitlab-ci.yml` with a YAML / JSON fixture lint stage
  and a pytest stage.

### Known limitations

- Reference scores in `data/code/reference_scores.yaml` are placeholders
  with `enforce: false`; populate with model-card numbers before relying
  on the warning system.
- Long-context retrieval (NIAH) suite is specified in
  `docs/extensions-spec.md` but not yet implemented.
- TRT-LLM and vLLM backends fall through to the stub adapter.

[Unreleased]: https://github.com/istanek/llm-benchmark/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/istanek/llm-benchmark/releases/tag/v0.2.1
[0.2.0]: https://github.com/istanek/llm-benchmark/releases/tag/v0.2.0
[0.1.0]: https://github.com/istanek/llm-benchmark/releases/tag/v0.1.0
