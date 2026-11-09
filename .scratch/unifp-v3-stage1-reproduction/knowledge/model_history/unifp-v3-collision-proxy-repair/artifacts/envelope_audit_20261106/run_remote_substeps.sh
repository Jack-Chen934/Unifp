#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03/source/UniFP_v3
DIAGNOSTIC_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/failure_visualization_20261106
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tools:/mnt/data_8THDD/myl_unifp/isaacgym/python:$DIAGNOSTIC_ROOT"
export LD_LIBRARY_PATH=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib
cd "$SOURCE_ROOT"
exec /mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python -u \
  "$DIAGNOSTIC_ROOT/contact_substeps_20261106/observe_contact_substeps.py" "$@"
