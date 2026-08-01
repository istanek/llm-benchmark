#!/usr/bin/env bash
# Put the CLI on $PATH under every name pyproject declares.
#
#     bash scripts/install-launchers.sh [target-dir]
#
# Default target is ~/.local/bin. Copies the launcher once and symlinks the
# remaining names to it, so `basename $0` can pick the right entry point.
# Re-run after moving the repo, or set LLM_BENCH_REPO instead.

set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$HOME/.local/bin}"
LAUNCHER="$TARGET/llm-bench"

mkdir -p "$TARGET"
install -m 755 "$REPO/scripts/llm-bench-launcher" "$LAUNCHER"

# Bake the checkout this was installed from into the installed copy, so moving
# the repo needs a re-run of this script and nothing in ~/.bashrc. An explicit
# LLM_BENCH_REPO still wins at runtime.
python3 - "$LAUNCHER" "$REPO" <<'PY'
import re, sys
path, repo = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
patched, count = re.subn(
    r'^REPO="\$\{LLM_BENCH_REPO:-[^}]*\}"$',
    f'REPO="${{LLM_BENCH_REPO:-{repo}}}"',
    text,
    count=1,
    flags=re.MULTILINE,
)
if count != 1:
    sys.exit("install-launchers.sh: could not find the REPO default line in the launcher")
open(path, "w", encoding="utf-8").write(patched)
PY

# spark-* are the pre-rename names. A previous `pip install -e .` left broken
# copies of them in ~/.local/bin importing the old module path, so these are
# overwritten deliberately rather than skipped.
for name in llm_benchmark llm-benchmark spark-bench spark-benchmark spark_benchmark; do
    ln -sf "$LAUNCHER" "$TARGET/$name"
done

echo "installed into $TARGET:"
echo "  llm-bench, spark-bench                 -> commands (run, compare, report, ...)"
echo "  llm_benchmark, llm-benchmark,"
echo "  spark_benchmark, spark-benchmark       -> interactive shell"
echo
echo "repo: $REPO"
case ":$PATH:" in
    *":$TARGET:"*) ;;
    *) echo "WARNING: $TARGET is not on your PATH — add it to ~/.bashrc." ;;
esac
