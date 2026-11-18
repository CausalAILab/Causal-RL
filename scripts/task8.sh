#!/usr/bin/env bash
set -euo pipefail
task8_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TASK8_PYTHON:-}" ]]; then
  task8_python="$TASK8_PYTHON"
elif [[ -x "$task8_root/.venv/bin/python" ]]; then
  task8_python="$task8_root/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  task8_python="$VIRTUAL_ENV/bin/python"
else
  printf '%s\n' 'Activate a virtual environment or set TASK8_PYTHON.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${TASK8_CACHE_HOME:-$task8_root/runs/task8/.cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
exec "$task8_python" "$task8_root/experiments/task8/run.py" "$@"
