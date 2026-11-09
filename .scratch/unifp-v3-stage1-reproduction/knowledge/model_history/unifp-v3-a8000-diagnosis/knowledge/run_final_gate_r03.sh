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
"$PYTHON_BIN" "$VALIDATION_ROOT/verify_r03_runs.py" --validation-root "$VALIDATION_ROOT" > "$VALIDATION_ROOT/verify_r03_runs.log" 2>&1
"$PYTHON_BIN" "$VALIDATION_ROOT/run_capacity_formal_r03.py" --reproduction_part A --run_purpose formal --seed 1 --num_envs 4096 --max_iterations 8000 --output_dir "$VALIDATION_ROOT/capacity4096" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/capacity4096.console.log" 2>&1
"$PYTHON_BIN" "$VALIDATION_ROOT/build_r03_start_evidence.py" --validation-root "$VALIDATION_ROOT" > "$VALIDATION_ROOT/build_r03_start_evidence.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.preflight_v3 --evidence "$VALIDATION_ROOT/start_evidence.json" --config_snapshot "$VALIDATION_ROOT/formal_config_snapshot.json" --runtime "$VALIDATION_ROOT/formal_runtime.json" --manifest "$PROJECT_ROOT/configs/v3_a_initial_v1.json" --output "$VALIDATION_ROOT/start_gate_recomputed.json" > "$VALIDATION_ROOT/start_gate_recomputed.log" 2>&1
