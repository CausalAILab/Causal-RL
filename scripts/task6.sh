#!/usr/bin/env bash
set -euo pipefail
task6_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TASK6_PYTHON:-}" ]]; then
  task6_python="$TASK6_PYTHON"
elif [[ -x "$task6_root/.venv/bin/python" ]]; then
  task6_python="$task6_root/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  task6_python="$VIRTUAL_ENV/bin/python"
else
  printf '%s\n' 'Activate a virtual environment or set TASK6_PYTHON.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${TASK6_CACHE_HOME:-$task6_root/runs/task6/.cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
exec "$task6_python" "$task6_root/experiments/task6/run.py" "$@"
