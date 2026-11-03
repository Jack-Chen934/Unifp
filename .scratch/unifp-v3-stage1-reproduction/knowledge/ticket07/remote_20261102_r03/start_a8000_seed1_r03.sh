#!/usr/bin/env bash
set -euo pipefail

# 仅在用户决定正式启动后执行；本文件的创建不代表训练已启动。
VALIDATION_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r03
PROJECT_ROOT="$VALIDATION_ROOT/UniFP_v3"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
TRAINING_OUTPUT=/mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_a_initial_v1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUTF8=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:/mnt/data_8THDD/myl_unifp/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_ROOT"

# 正式入口会重算当次身份、核验四阶段报告，再创建仿真；不复用容量短测权重。
exec "$PYTHON_BIN" -m legged_gym.scripts.train_v3 \
  --reproduction_part A --run_purpose formal \
  --seed 1 --num_envs 4096 --max_iterations 8000 \
  --start_evidence "$VALIDATION_ROOT/start_evidence.json" \
  --manifest "$PROJECT_ROOT/configs/v3_a_initial_v1.json" \
  --output_dir "$TRAINING_OUTPUT" \
  --sim_device cuda:0 --rl_device cuda:0 --headless
