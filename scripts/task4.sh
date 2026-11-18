#!/usr/bin/env bash
set -euo pipefail
task4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TASK4_PYTHON:-}" ]]; then
  task4_python="$TASK4_PYTHON"
elif [[ -x "$task4_root/.venv/bin/python" ]]; then
  task4_python="$task4_root/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  task4_python="$VIRTUAL_ENV/bin/python"
else
  printf '%s\n' 'Activate a virtual environment or set TASK4_PYTHON.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${TASK4_CACHE_HOME:-$task4_root/runs/task4/.cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
exec "$task4_python" "$task4_root/experiments/task4/run.py" "$@"
