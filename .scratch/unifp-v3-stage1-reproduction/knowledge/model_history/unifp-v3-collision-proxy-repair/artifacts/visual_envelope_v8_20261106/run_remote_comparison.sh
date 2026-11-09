#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03/source/UniFP_v3
DIAGNOSTIC_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/failure_visualization_20261106
EVIDENCE_ROOT="$DIAGNOSTIC_ROOT/visual_envelope_v8_20261106"
CHECKPOINT=/mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_001400.pt
V8_URDF="$EVIDENCE_ROOT/asset/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
R3_URDF="$DIAGNOSTIC_ROOT/grouped_r3_acceptance/asset/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tools:/mnt/data_8THDD/myl_unifp/isaacgym/python"
export LD_LIBRARY_PATH="/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SOURCE_ROOT"
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" -u "$DIAGNOSTIC_ROOT/replay_v3.py" --checkpoint "$CHECKPOINT" --output_dir "$EVIDENCE_ROOT/v8_seed1001_20s" --num_envs 128 --seconds 20 --seed 1001 --action_mode sample --scenario S1 --asset_override "$V8_URDF" > "$EVIDENCE_ROOT/v8_seed1001_20s.log" 2>&1
printf 'V8 seed1001 128环境×20秒完成，退出0\n'
# 各seed独占一张同型号A6000，三种资产在同一seed内依次使用同一GPU。
# 仅并行独立的零更新测试，不将共享机器的耗时用于性能比较。
run_frozen_seed() {
  local replay_seed="$1" diagnostic_gpu="$2" variant output
  local -a asset_arguments
  for variant in v6 r3 v8; do
    asset_arguments=()
    if [[ "$variant" == r3 ]]; then
      asset_arguments=(--asset_override "$R3_URDF")
    elif [[ "$variant" == v8 ]]; then
      asset_arguments=(--asset_override "$V8_URDF")
    fi
    output="$EVIDENCE_ROOT/${variant}_seed${replay_seed}_60s"
    CUDA_VISIBLE_DEVICES="$diagnostic_gpu" "$PYTHON_BIN" -u "$DIAGNOSTIC_ROOT/replay_v3.py" --checkpoint "$CHECKPOINT" --output_dir "$output" --num_envs 128 --seconds 60 --seed "$replay_seed" --action_mode sample --scenario S1 "${asset_arguments[@]}" > "${output}.log" 2>&1
    printf '%s seed%s GPU%s 128环境×60秒完成，退出0\n' "$variant" "$replay_seed" "$diagnostic_gpu"
  done
}
run_frozen_seed 1001 0 &
seed1001_pid=$!
run_frozen_seed 1002 1 &
seed1002_pid=$!
run_frozen_seed 1003 3 &
seed1003_pid=$!
wait "$seed1001_pid"
wait "$seed1002_pid"
wait "$seed1003_pid"
printf '全部九组零更新对照完成\n'
