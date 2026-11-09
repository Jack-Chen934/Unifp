#!/usr/bin/env bash
set -euo pipefail
VALIDATION_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03
PROJECT_ROOT="$VALIDATION_ROOT/source/UniFP_v3"
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/tools:/mnt/data_8THDD/myl_unifp/isaacgym/python"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -u "$VALIDATION_ROOT/train_observed.py" --reproduction_part A --run_purpose formal --seed 1 --num_envs 4096 --max_iterations 8000 --output_dir /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_20261104 --start_evidence "$VALIDATION_ROOT/start_evidence.json" --manifest "$PROJECT_ROOT/configs/v3_a_initial_v1.json" --sim_device cuda:0 --rl_device cuda:0 --headless
