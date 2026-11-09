#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02/source/UniFP_v3
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:/mnt/data_8THDD/myl_unifp/isaacgym/python"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GIT_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/evaluation_provenance/a8000_collision_r02_20261104.git
export GIT_WORK_TREE="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m legged_gym.scripts.evaluate_v3 \
 --reproduction_part A --run_purpose evaluation --seed 1001 --num_envs 32 \
 --checkpoint /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_collision_v1/checkpoints/model_008000.pt \
 --manifest "$PROJECT_ROOT/configs/v3_a_initial_v1.json" \
 --output_dir /mnt/data_8THDD/myl_unifp_v3_stage1/evaluations/a8000_seed1_v3_collision_v1_20261104_r02 \
 --sim_device cuda:0 --rl_device cuda:0 --headless
