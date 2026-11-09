#!/usr/bin/env bash
set -euo pipefail

# 此脚本只准备，不在优化验收期间执行；新资产seed1正式从零开始。
VALIDATION_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02
PROJECT_ROOT="$VALIDATION_ROOT/source/UniFP_v3"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export CUDA_VISIBLE_DEVICES=1
export PYTHONUTF8=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:/mnt/data_8THDD/myl_unifp/isaacgym/python"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_ROOT"

# 固定GPU1，与启动证据的runtime一致；更换设备必须重建对应证据。
# 未传--resume；容量短测权重和旧资产权重均不用于本次正式起跑。
exec "$PYTHON_BIN" -m legged_gym.scripts.train_v3 \
  --reproduction_part A --run_purpose formal \
  --seed 1 --num_envs 4096 --max_iterations 8000 \
  --start_evidence "$VALIDATION_ROOT/start_evidence.json" \
  --manifest "$PROJECT_ROOT/configs/v3_a_initial_v1.json" \
  --output_dir /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_collision_v1 \
  --sim_device cuda:0 --rl_device cuda:0 --headless
