#!/usr/bin/env bash
set -euo pipefail

# 仅短测：正式配置容量只执行1轮，不启动A8000，也不操作旧训练。
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

# 先逐字节核对源码，再在这一固定版本完成全部验证。
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import hashlib
import json
root = Path.cwd()
expected = json.loads((root.parents[1] / 'local_cpu_tests.json').read_text())['source_manifest']
actual = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
          for path in list((root / 'legged_gym').rglob('*.py')) + [root / 'SOURCE_BASELINE.json']}
assert actual == expected, '远程源码与本机单测源码不一致'
PY
ASSET_PATH="$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v1/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$ASSET_PATH" --output "$VALIDATION_ROOT/collision_candidate_nominal32.json" --num-envs 32 --steps 250 --nominal > "$VALIDATION_ROOT/collision_candidate_nominal32.console.log" 2>&1
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$ASSET_PATH" --output "$VALIDATION_ROOT/collision_candidate_nominal4.json" --num-envs 4 --steps 250 --nominal > "$VALIDATION_ROOT/collision_candidate_nominal4.console.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 512 --max_iterations 10 --output_dir "$VALIDATION_ROOT/run512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/run512.console.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 512 --max_iterations 1 --resume "$VALIDATION_ROOT/run512/checkpoints/model_000010.pt" --output_dir "$VALIDATION_ROOT/resume512" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/resume512.console.log" 2>&1
"$PYTHON_BIN" tools/benchmark_collision_training.py --source-root "$PROJECT_ROOT" --timing-output "$VALIDATION_ROOT/timing4096.json" --reproduction_part A --run_purpose development --seed 1 --num_envs 4096 --max_iterations 100 --output_dir "$VALIDATION_ROOT/run4096" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/run4096.console.log" 2>&1
"$PYTHON_BIN" -m unittest discover -s tests -p test_v3_rewards.py -v > "$VALIDATION_ROOT/reward_unit.console.log" 2>&1
"$PYTHON_BIN" -c 'import json; from tools.check_v3_desktop_runtime import check_runtime; print(json.dumps(check_runtime(), ensure_ascii=False))' > "$VALIDATION_ROOT/nominal32.console.log" 2>&1
"$PYTHON_BIN" -c 'import json; from tools.check_v3_reward_runtime import check_runtime; print(json.dumps(check_runtime(), ensure_ascii=False))' > "$VALIDATION_ROOT/randomization8.console.log" 2>&1
"$PYTHON_BIN" -m legged_gym.scripts.train_v3 --reproduction_part A --run_purpose development --seed 1 --num_envs 32 --max_iterations 1 --output_dir "$VALIDATION_ROOT/smoke32" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/smoke32.console.log" 2>&1
"$PYTHON_BIN" "$VALIDATION_ROOT/run_capacity_formal.py" --reproduction_part A --run_purpose formal --seed 1 --num_envs 4096 --max_iterations 8000 --output_dir "$VALIDATION_ROOT/capacity4096" --sim_device cuda:0 --rl_device cuda:0 --headless > "$VALIDATION_ROOT/capacity4096.console.log" 2>&1
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$VALIDATION_ROOT/gpu_after.csv"
