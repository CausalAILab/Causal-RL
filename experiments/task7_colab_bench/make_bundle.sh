#!/usr/bin/env bash
# Pack the current working-tree source needed by the Task 7 speed test into task7_colab_bench.zip.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl="$(cd -- "$here/../.." && pwd)"
gym="$(cd -- "$rl/../Causal-Gymnasium" && pwd)"
stage="$(mktemp -d)"; trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/task7_colab_bench/code" "$stage/task7_colab_bench/bench"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$rl/causal_rl" "$stage/task7_colab_bench/code/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' "$gym/causal_gym" "$stage/task7_colab_bench/code/"
cp "$here/bench_task7.py" "$here/run_suite.py" "$here/summarize_bench.py" "$here/requirements_colab.txt" "$stage/task7_colab_bench/bench/"
{
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Causal-RL $(git -C "$rl" rev-parse HEAD) dirty_files=$(git -C "$rl" status --porcelain -- causal_rl | wc -l | tr -d ' ')"
  echo "Causal-Gymnasium $(git -C "$gym" rev-parse HEAD) dirty_files=$(git -C "$gym" status --porcelain -- causal_gym | wc -l | tr -d ' ')"
  echo "source sha256:"
  (cd "$stage/task7_colab_bench" && find code bench -type f | LC_ALL=C sort | xargs shasum -a 256)
} > "$stage/task7_colab_bench/MANIFEST.txt"
rm -f "$here/task7_colab_bench.zip"
(cd "$stage" && zip -qr "$here/task7_colab_bench.zip" task7_colab_bench)
echo "wrote $here/task7_colab_bench.zip ($(du -h "$here/task7_colab_bench.zip" | cut -f1))"
