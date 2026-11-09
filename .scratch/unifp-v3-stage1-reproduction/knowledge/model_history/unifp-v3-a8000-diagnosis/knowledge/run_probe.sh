#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02/source/UniFP_v3
DIAG_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/a8000_20261104
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:/mnt/data_8THDD/myl_unifp/isaacgym/python"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec /mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python "$DIAG_ROOT/$1" --source-root "$PROJECT_ROOT" "${@:2}"
