#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03/source/UniFP_v3
DIAGNOSTIC_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/failure_visualization_20261106
EVIDENCE_ROOT="$DIAGNOSTIC_ROOT/grouped_r3_acceptance"
CANDIDATE_URDF="$EVIDENCE_ROOT/asset/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export LD_LIBRARY_PATH=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tools:/mnt/data_8THDD/myl_unifp/isaacgym/python"
cd "$SOURCE_ROOT"
for motion_sign in 1 -1; do
  "$PYTHON_BIN" -u "$EVIDENCE_ROOT/probe_wheels.py" --source-root "$SOURCE_ROOT" --asset "$CANDIDATE_URDF" --motion-sign "$motion_sign" --output "$EVIDENCE_ROOT/wheels_sign_${motion_sign}.json" > "$EVIDENCE_ROOT/wheels_sign_${motion_sign}.log" 2>&1
  printf '轮系 motion_sign=%s 完成，退出0\n' "$motion_sign"
done
for scenario in S1 training; do
  bash "$DIAGNOSTIC_ROOT/run_replay_v3.sh" --checkpoint /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_001400.pt --output_dir "$EVIDENCE_ROOT/sample_${scenario}_20s" --num_envs 128 --seconds 20 --seed 1001 --action_mode sample --scenario "$scenario" --asset_override "$CANDIDATE_URDF" > "$EVIDENCE_ROOT/sample_${scenario}_20s.log" 2>&1
  printf '128环境 %s 20秒采样完成，退出0\n' "$scenario"
done
