#!/usr/bin/env bash
set -euo pipefail
task7_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TASK7_PYTHON:-}" ]]; then
  task7_python="$TASK7_PYTHON"
elif [[ -x "$task7_root/.venv/bin/python" ]]; then
  task7_python="$task7_root/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  task7_python="$VIRTUAL_ENV/bin/python"
else
  printf '%s\n' 'Activate a virtual environment or set TASK7_PYTHON.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg SDL_VIDEODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="${TASK7_CACHE_HOME:-$task7_root/runs/task7/.cache}"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
exec "$task7_python" "$task7_root/experiments/task7/run.py" "$@"
