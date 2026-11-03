#!/usr/bin/env bash
set -euo pipefail

VALIDATION_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r01
PROJECT_ROOT="$VALIDATION_ROOT/UniFP_v3"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export CUDA_VISIBLE_DEVICES=0
export PYTHONUTF8=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$PROJECT_ROOT:/mnt/data_8THDD/myl_unifp/isaacgym/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c 'import isaacgym; import runpy; from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg; V3PosForceCfg.sim.physx.max_gpu_contact_pairs = 2**25; runpy.run_module("legged_gym.scripts.train_v3", run_name="__main__")' --reproduction_part A --run_purpose preflight --seed 0 --num_envs 4096 --max_iterations 1 --output_dir "$VALIDATION_ROOT/runs/capacity4096_pairs25" --sim_device cuda:0 --rl_device cuda:0 --headless
