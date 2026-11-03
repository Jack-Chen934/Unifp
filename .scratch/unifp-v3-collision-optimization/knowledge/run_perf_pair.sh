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

# 同一物理GPU、源码、参数和seed，独立进程顺序对照。
BASELINE_ASSET=/mnt/data_8THDD/myl_unifp_v3_stage1/UniFP_v3/resources/robots/v3_a1_desktop/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf
CANDIDATE_ASSET="$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v1/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$BASELINE_ASSET" --output "$VALIDATION_ROOT/collision_baseline_randomized512.json" --num-envs 512 --steps 100 > "$VALIDATION_ROOT/collision_baseline_randomized512.console.log" 2>&1
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$CANDIDATE_ASSET" --output "$VALIDATION_ROOT/collision_candidate_randomized512.json" --num-envs 512 --steps 100 > "$VALIDATION_ROOT/collision_candidate_randomized512.console.log" 2>&1
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$BASELINE_ASSET" --output "$VALIDATION_ROOT/collision_baseline_physics512.json" --num-envs 512 --steps 100 --physics-only > "$VALIDATION_ROOT/collision_baseline_physics512.console.log" 2>&1
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$CANDIDATE_ASSET" --output "$VALIDATION_ROOT/collision_candidate_physics512.json" --num-envs 512 --steps 100 --physics-only > "$VALIDATION_ROOT/collision_candidate_physics512.console.log" 2>&1
"$PYTHON_BIN" "$VALIDATION_ROOT/check_old_resume.py"
"$PYTHON_BIN" "$VALIDATION_ROOT/summarize_validation.py" > "$VALIDATION_ROOT/summary.console.log" 2>&1
