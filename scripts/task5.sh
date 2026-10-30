#!/usr/bin/env bash
set -euo pipefail
task5_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TASK5_PYTHON:-}" ]]; then
  task5_python="$TASK5_PYTHON"
elif [[ -x "$task5_root/.venv/bin/python" ]]; then
  task5_python="$task5_root/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  task5_python="$VIRTUAL_ENV/bin/python"
else
  printf '%s\n' 'No virtual environment found. Create/activate one or set TASK5_PYTHON.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$task5_root/runs/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
exec "$task5_python" "$task5_root/experiments/task5.py" "$@"
