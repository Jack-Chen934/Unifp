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

if [[ "${1:-all}" == all ]]; then
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$VALIDATION_ROOT/evidence/gpu_before.csv"
echo '开始奖励张量单测'
"$PYTHON_BIN" -m unittest discover -s tests -p test_v3_rewards.py -v > "$VALIDATION_ROOT/evidence/reward_unit.console.log" 2>&1
echo '开始32环境名义物理与局部reset检查'
"$PYTHON_BIN" -c 'import json; from tools.check_v3_desktop_runtime import check_runtime; print(json.dumps(check_runtime(), ensure_ascii=False))' > "$VALIDATION_ROOT/evidence/nominal32.console.log" 2>&1
echo '开始8环境随机化与push检查'
"$PYTHON_BIN" -c 'import json; from tools.check_v3_reward_runtime import check_runtime; print(json.dumps(check_runtime(), ensure_ascii=False))' > "$VALIDATION_ROOT/evidence/randomization8.console.log" 2>&1
fi
echo '开始32环境1轮更新'
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose preflight --seed 0 --num_envs 32 --max_iterations 1 --output_dir "$VALIDATION_ROOT/runs/smoke32" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/evidence/smoke32.console.log" 2>&1
echo '开始512环境10轮更新'
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose preflight --seed 0 --num_envs 512 --max_iterations 10 --output_dir "$VALIDATION_ROOT/runs/train512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/evidence/train512.console.log" 2>&1
echo '开始512环境跨进程续1轮'
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose preflight --seed 0 --num_envs 512 --max_iterations 1 --resume "$VALIDATION_ROOT/runs/train512/checkpoints/model_000010.pt" --output_dir "$VALIDATION_ROOT/runs/resume512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/evidence/resume512.console.log" 2>&1
echo '开始4096环境候选容量1轮预检'
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose preflight --seed 0 --num_envs 4096 --max_iterations 1 --output_dir "$VALIDATION_ROOT/runs/capacity4096" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/evidence/capacity4096.console.log" 2>&1
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv > "$VALIDATION_ROOT/evidence/gpu_after.csv"
echo '分层验证命令全部正常退出；尚需核查原始输出、检查点与物理失败统计'
