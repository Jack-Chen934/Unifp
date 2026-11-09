#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r03/source/UniFP_v3
EVIDENCE_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/failure_visualization_20261106/visual_envelope_v8_20261106
CANDIDATE_URDF="$EVIDENCE_ROOT/asset/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
PYTHON_BIN=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export LD_LIBRARY_PATH=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib
export PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/tools:/mnt/data_8THDD/myl_unifp/isaacgym/python"
cd "$SOURCE_ROOT"
"$PYTHON_BIN" -u tools/probe_training_collision_runtime.py --source-root "$SOURCE_ROOT" --asset "$CANDIDATE_URDF" --num-envs 4 --steps 250 --nominal --output "$EVIDENCE_ROOT/remote_runtime.json" > "$EVIDENCE_ROOT/remote_runtime.log" 2>&1
printf '远程静置／轮系／小幅臂动作、reset 完成，退出0\n'
for motion_sign in 1 -1; do
  "$PYTHON_BIN" -u "$EVIDENCE_ROOT/probe_wheels.py" --source-root "$SOURCE_ROOT" --asset "$CANDIDATE_URDF" --motion-sign "$motion_sign" --output "$EVIDENCE_ROOT/wheels_sign_${motion_sign}.json" > "$EVIDENCE_ROOT/wheels_sign_${motion_sign}.log" 2>&1
  printf '轮系 motion_sign=%s 完成，退出0\n' "$motion_sign"
done
