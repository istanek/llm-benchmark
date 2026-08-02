#!/usr/bin/env python3
"""Build a semantically identical, textually novel variant of MBPP.

    python3 scripts/build_mbpp_mutants.py

MBPP and HumanEval have been public for years and are in the training data of
every model measured here. A high score can mean the model solved the problem
or that it recalled it, and nothing in the suite currently tells the two apart.

This writes a mutated fixture whose tasks are the same problems in different
words:

- the entry point is renamed through a synonym map, keeping the semantic hint
  the name carries (``similar_elements`` -> ``shared_items``) rather than
  erasing it, since ``func_1`` would make every task harder for everyone and
  confound the measurement it is meant to enable;
- the prose is paraphrased through fixed templates;
- the example call embedded in the prompt gets fresh arguments, with the
  expected value **recomputed by executing the canonical solution** — so the
  example is novel and still true.

Hidden tests keep their original inputs. The model never sees them, so
mutating them would add no novelty, only risk.

Every mutant is validated the only way that counts: its canonical solution is
run against its tests in this repo's sandbox. A task that fails is dropped,
never shipped on the assumption that the transformation was safe.

The measurement this enables is the *difference* between a model's score on
MBPP and on this set. Both are the same problems, so a large drop is evidence
of recall rather than ability. The absolute score here is not the point and
should not be quoted on its own — some drop is expected for every model,
because unfamiliar naming is genuinely a little harder.
"""

from __future__ import annotations

import ast
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_benchmark.code_generation import (  # noqa: E402
    _build_program,
    extract_code,
    load_mbpp_suite,
    sandbox_run,
)

SOURCE = REPO_ROOT / "data" / "code" / "code_generation_mbpp_v1.json"
TARGET = REPO_ROOT / "data" / "code" / "code_generation_mbpp_mutated_v1.json"

# Synonyms for the vocabulary MBPP entry points are built from. Chosen to keep
# the meaning the name conveys: the goal is a name the model has not seen for
# this problem, not a name that hides what the problem is.
SYNONYMS = {
    "find": "locate", "get": "fetch", "count": "tally", "check": "verify",
    "remove": "drop", "sum": "total", "add": "append", "convert": "translate",
    "calculate": "compute", "extract": "pull", "sort": "order", "merge": "combine",
    "split": "divide", "replace": "substitute", "reverse": "invert", "max": "largest",
    "min": "smallest", "list": "sequence", "lists": "sequences", "string": "text",
    "strings": "texts", "array": "vector", "arrays": "vectors", "number": "value",
    "numbers": "values", "elements": "items", "element": "item", "tuple": "grouping",
    "tuples": "groupings", "dict": "mapping", "char": "character", "chars": "characters",
    "word": "term", "words": "terms", "digit": "numeral", "digits": "numerals",
    "first": "initial", "last": "final", "same": "identical", "similar": "shared",
    "given": "supplied", "length": "size", "index": "position", "value": "amount",
    "product": "multiple", "even": "divisible", "odd": "nondivisible",
    "positive": "nonnegative", "unique": "distinct", "common": "mutual",
    "range": "span", "matrix": "grid", "nested": "inner", "front": "leading",
    "rear": "trailing", "upper": "capital", "lower": "small", "empty": "blank",
    "pair": "duo", "pairs": "duos", "consecutive": "successive", "occurrence": "appearance",
    "occurrences": "appearances", "frequency": "rate", "position": "slot",
    "middle": "central", "second": "runnerup", "third": "tertiary",
}

PARAPHRASES = [
    ("write a function to ", "Implement a Python function that will "),
    ("write a function to ", "Define a function which should "),
    ("write a python function to ", "Implement a function that will "),
    ("write a python function to ", "Define a Python function which should "),
    ("write a function that ", "Implement a Python function that "),
    ("write a python function that ", "Implement a function that "),
    ("write function to ", "Implement a function that will "),
]


def rename_entry_point(name: str, rng: random.Random) -> str:
    """Rename through synonyms, with fallbacks that never return the original."""
    parts = name.split("_")
    renamed = [SYNONYMS.get(part.lower(), part) for part in parts]
    if renamed != parts:
        return "_".join(renamed)
    # No word had a synonym. Reordering keeps the words (and the hint) while
    # changing the token the model would have memorised.
    if len(parts) > 1:
        return "_".join(parts[1:] + parts[:1])
    return rng.choice(["compute_", "derive_", "produce_"]) + name


def paraphrase_prose(prompt_body: str) -> str:
    """Rewrite the leading instruction; leave the rest of the sentence alone."""
    lowered = prompt_body.lstrip()
    for prefix, replacement in PARAPHRASES:
        if lowered.lower().startswith(prefix):
            return replacement + lowered[len(prefix):]
    return prompt_body


def perturb_literal(value: Any, rng: random.Random) -> Any:
    """A different value of the same shape.

    Type-preserving on purpose: the mutant has to stay solvable by the same
    algorithm, so a list of ints stays a list of ints of similar magnitude.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        delta = rng.choice([1, 2, 3])
        return value + delta if value >= 0 else value - delta
    if isinstance(value, float):
        return round(value + rng.choice([0.5, 1.5, 2.0]), 4)
    if isinstance(value, str):
        if not value:
            return value
        pool = "abcdefghijklmnopqrstuvwxyz"
        chars = list(value)
        for index, char in enumerate(chars):
            if char.isalpha():
                replacement = rng.choice(pool)
                chars[index] = replacement.upper() if char.isupper() else replacement
                break
        return "".join(chars)
    if isinstance(value, list):
        return [perturb_literal(item, rng) for item in value]
    if isinstance(value, tuple):
        return tuple(perturb_literal(item, rng) for item in value)
    if isinstance(value, set):
        return {perturb_literal(item, rng) for item in value}
    if isinstance(value, dict):
        return {key: perturb_literal(item, rng) for key, item in value.items()}
    return value


def keep_needles_in_haystacks(original: tuple[Any, ...], mutated: list[Any]) -> list[Any]:
    """Repair single-character arguments that were meant to occur in a string.

    ``remove_Occ("hello", "l")`` perturbed naively becomes
    ``remove_Occ("gello", "s")`` — still true once the expected value is
    recomputed, and useless as an example, because the character is no longer
    in the string. The example stops illustrating the task, which is a
    confound: it makes the mutant harder for reasons that have nothing to do
    with whether the model memorised the original.
    """
    repaired = list(mutated)
    for index, value in enumerate(original):
        if not (isinstance(value, str) and len(value) == 1):
            continue
        for other_index, other in enumerate(original):
            if other_index == index or not isinstance(other, str) or len(other) <= 1:
                continue
            if value not in other:
                continue
            haystack = repaired[other_index]
            if isinstance(haystack, str) and haystack:
                # Same relative position as in the original, so the example
                # keeps its shape (first/last occurrence, and so on).
                position = min(other.index(value), len(haystack) - 1)
                repaired[index] = haystack[position]
    return repaired


def is_degenerate(original_args: tuple[Any, ...], original_expected: Any, new_args: tuple[Any, ...], new_expected: Any) -> bool:
    """True when the mutation turned the example into a no-op.

    The check is relative: an example whose answer *is* one of its inputs is
    fine if the original was like that too. It is only suspicious when the
    mutation created that situation.
    """
    if original_expected is _UNKNOWN:
        return False
    was_identity = any(original_expected == arg for arg in original_args)
    is_identity = any(new_expected == arg for arg in new_args)
    if is_identity and not was_identity:
        return True
    # An empty result where the original produced something is the other
    # common way a perturbed input stops exercising the function.
    empty_now = new_expected in ([], "", (), {}, set(), 0)
    empty_before = original_expected in ([], "", (), {}, set(), 0)
    return empty_now and not empty_before


_UNKNOWN = object()


def evaluate_canonical(canonical: str, entry_point: str, args: tuple[Any, ...]) -> tuple[bool, Any]:
    """Run the canonical solution on *args* in a subprocess and return its result.

    Out of process because a fixture solution can loop forever or exhaust
    memory; the generator must not be the thing that hangs.
    """
    program = (
        f"{canonical}\n\n"
        "import json, sys\n"
        f"result = {entry_point}(*{args!r})\n"
        "sys.stdout.write(repr(result))\n"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=10
        )
    except subprocess.SubprocessError:
        return False, None
    if completed.returncode != 0:
        return False, None
    try:
        return True, ast.literal_eval(completed.stdout)
    except (ValueError, SyntaxError):
        return False, None


def mutate_example(
    prompt: str,
    *,
    name_in_prompt: str,
    canonical: str,
    canonical_entry: str,
    rng: random.Random,
) -> tuple[str, bool]:
    """Give the prompt's example call fresh arguments and a true expected value.

    Returns (prompt, mutated). When the assert is not of the simple
    ``f(args) == literal`` shape, or the canonical solution rejects the new
    arguments, the example keeps its original values and only the name changes
    — a partial mutation is still a mutation, and a wrong example would be far
    worse than a familiar one.
    """
    match = re.search(r"^assert\s+(.+)$", prompt, flags=re.MULTILINE)
    if not match:
        return prompt, False
    expression = match.group(1).strip()
    # The prompt has already been renamed, so the call to match is the new name;
    # the canonical solution is still the original, so it is executed under the
    # original one. Conflating the two silently produced zero mutated examples.
    simple = re.fullmatch(re.escape(name_in_prompt) + r"\((.*)\)\s*==\s*(.+)", expression, flags=re.DOTALL)
    if not simple:
        return prompt, False

    try:
        call = ast.parse(f"{name_in_prompt}({simple.group(1)})", mode="eval").body
        args = tuple(ast.literal_eval(arg) for arg in call.args)  # type: ignore[attr-defined]
    except (SyntaxError, ValueError):
        return prompt, False

    try:
        original_expected = ast.literal_eval(simple.group(2).strip())
    except (ValueError, SyntaxError):
        original_expected = _UNKNOWN

    for _ in range(8):
        candidate = [perturb_literal(arg, rng) for arg in args]
        new_args = tuple(keep_needles_in_haystacks(args, candidate))
        if new_args == args:
            continue
        ok, expected = evaluate_canonical(canonical, canonical_entry, new_args)
        if not ok:
            continue
        if is_degenerate(args, original_expected, new_args, expected):
            continue
        rendered_args = ", ".join(repr(arg) for arg in new_args)
        new_assert = f"assert {name_in_prompt}({rendered_args}) == {expected!r}"
        return prompt.replace(match.group(0), new_assert), True
    return prompt, False


def rename_everywhere(text: str, old: str, new: str) -> str:
    """Rename every occurrence — for code, where every occurrence is the symbol."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def rename_call_sites(text: str, old: str, new: str) -> str:
    """Rename only where the name is being called.

    Entry points like ``sequence``, ``power`` and ``search`` are ordinary
    English words, and renaming them everywhere mangled the problem statement:
    "the nth number in the newman conway sequence" became "... newman conway
    compute_sequence". The prose describes the problem and must survive; only
    the call in the example assert is the symbol.
    """
    return re.sub(rf"\b{re.escape(old)}\b(?=\s*\()", new, text)


def main() -> int:
    suite = load_mbpp_suite(REPO_ROOT)
    source = json.loads(SOURCE.read_text())

    mutants: list[dict[str, Any]] = []
    stats = {"renamed": 0, "example_mutated": 0, "dropped": 0, "paraphrased": 0}

    for index, task in enumerate(suite.tasks):
        rng = random.Random(f"mbpp-mutant::{task.task_id}")
        entry_point = str(task.metadata.get("entry_point") or "")
        canonical = str(task.metadata.get("canonical_solution") or "")
        tests = str(task.metadata.get("tests") or "")
        if not entry_point or not canonical or not tests:
            stats["dropped"] += 1
            continue

        new_name = rename_entry_point(entry_point, rng)
        if new_name == entry_point:
            stats["dropped"] += 1
            continue

        prompt = rename_call_sites(task.prompt, entry_point, new_name)
        before = prompt
        # splitlines() drops a trailing newline and "\n".join does not restore
        # it. That left prompts ending in `"""` with no line break, so a model
        # echoing the prompt before its answer produced `"""def f(...)` — an
        # unterminated docstring. The generator's own validation did not
        # simulate that path; the fixture canary did, and caught 33 tasks.
        trailing = "\n" if prompt.endswith("\n") else ""
        prompt = "\n".join(
            paraphrase_prose(line) if line.strip() and not line.startswith(('"""', "assert")) else line
            for line in prompt.splitlines()
        ) + trailing
        if prompt != before:
            stats["paraphrased"] += 1

        prompt, mutated_example = mutate_example(
            prompt,
            name_in_prompt=new_name,
            canonical=canonical,
            canonical_entry=entry_point,
            rng=rng,
        )
        stats["example_mutated"] += int(mutated_example)

        new_canonical = rename_everywhere(canonical, entry_point, new_name)
        new_tests = rename_everywhere(tests, entry_point, new_name)

        # Validate through the same path the fixture canary uses: the canonical
        # solution *preceded by the prompt*, as a model echoing the prompt would
        # produce, run through extract_code and the sandbox. Validating the
        # solution alone is a weaker check and passed 33 tasks that were broken.
        program = _build_program(
            extract_code(prompt + new_canonical, prompt, new_name),
            new_tests,
            prompt=prompt,
            entry_point=new_name,
        )
        result = sandbox_run(program, timeout_s=15.0, memory_limit_mb=1024)
        if not result.passed:
            stats["dropped"] += 1
            continue

        stats["renamed"] += 1
        mutants.append(
            {
                "task_id": task.task_id.replace("mbpp/", "mbpp-mut/"),
                "prompt": prompt,
                "context": task.context,
                "reference": task.reference,
                "tags": list(task.tags) + ["mutated"],
                "metadata": {
                    **task.metadata,
                    "benchmark": "mbpp_mutated",
                    "entry_point": new_name,
                    "tests": new_tests,
                    "canonical_solution": new_canonical,
                    "mutation": {
                        "source_task": task.task_id,
                        "original_entry_point": entry_point,
                        "example_arguments_recomputed": mutated_example,
                    },
                },
            }
        )

    fixture = {
        "name": "code_generation_mbpp_mutated_v1",
        "category": source.get("category", "code"),
        "version": "0.1.0",
        "description": (
            "Semantics-preserving mutation of MBPP sanitized, for contamination "
            "detection. Same problems, renamed entry points, paraphrased prose and "
            "recomputed example arguments. Read as a delta against the original set, "
            "never on its own."
        ),
        "notes": [
            "Generated by scripts/build_mbpp_mutants.py; regenerate rather than edit by hand.",
            "Every task's canonical solution was executed against its tests in this repo's "
            "sandbox before being written here. Tasks that failed were dropped.",
            "Hidden tests keep their original inputs: the model never sees them, so mutating "
            "them would add risk without adding novelty.",
            "A drop against the original MBPP score is evidence of recall rather than ability. "
            "Some drop is expected for every model — unfamiliar naming is genuinely harder — so "
            "the comparison worth making is between models, not against 100 %.",
            f"{stats['example_mutated']} of {len(mutants)} tasks also have fresh example "
            "arguments, with the expected value recomputed by running the canonical solution.",
        ],
        "tasks": mutants,
    }
    TARGET.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"kept      {len(mutants)} / {len(suite.tasks)}")
    print(f"dropped   {stats['dropped']} (canonical solution did not survive the mutation)")
    print(f"paraphrased prose      {stats['paraphrased']}")
    print(f"example args recomputed {stats['example_mutated']}")
    print(f"written   {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
