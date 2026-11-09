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
"$PYTHON_BIN" tools/check_v3_reset_tcp.py --output "$VALIDATION_ROOT/final_reset_tcp.json" > "$VALIDATION_ROOT/final_reset_tcp.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 32 --max_iterations 1 --output_dir "$VALIDATION_ROOT/smoke32" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/smoke32.console.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 512 --max_iterations 10 --output_dir "$VALIDATION_ROOT/run512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/run512.console.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 512 --max_iterations 1 --resume "$VALIDATION_ROOT/run512/checkpoints/model_000010.pt" --output_dir "$VALIDATION_ROOT/resume512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/resume512.console.log" 2>&1
"$PYTHON_BIN" tools/benchmark_collision_training.py --source-root "$PROJECT_ROOT" --timing-output "$VALIDATION_ROOT/timing4096.json" --reproduction_part A --run_purpose development --seed 1 --num_envs 4096 --max_iterations 20 --output_dir "$VALIDATION_ROOT/run4096" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/run4096.console.log" 2>&1
