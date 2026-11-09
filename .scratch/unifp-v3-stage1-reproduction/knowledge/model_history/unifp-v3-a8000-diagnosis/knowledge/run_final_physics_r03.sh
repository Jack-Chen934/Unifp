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
"$PYTHON_BIN" - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd()
expected=json.loads((root.parents[1]/'final_manifest.json').read_text())
for section in ('source_manifest','asset_manifest'):
    for name,digest in expected[section].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('源码与资产哈希核验通过')
PY
ASSET_PATH="$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf"
"$PYTHON_BIN" -m unittest discover -s tests -p test_v3_control_config.py -v > "$VALIDATION_ROOT/control_unit.log" 2>&1
"$PYTHON_BIN" -m unittest discover -s tests -p test_v3_rewards.py -v > "$VALIDATION_ROOT/rewards_unit.log" 2>&1
for motion_sign in 1 -1; do
    "$PYTHON_BIN" "$VALIDATION_ROOT/probe_wheels.py" --source-root "$PROJECT_ROOT" --output "$VALIDATION_ROOT/final_motion_sign${motion_sign}.json" --motion-sign "$motion_sign" > "$VALIDATION_ROOT/final_motion_sign${motion_sign}.log" 2>&1
done
"$PYTHON_BIN" "$VALIDATION_ROOT/check_motion_reports.py" "$VALIDATION_ROOT/final_motion_sign1.json" "$VALIDATION_ROOT/final_motion_sign-1.json" > "$VALIDATION_ROOT/final_motion_gate.json"
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$ASSET_PATH" --output "$VALIDATION_ROOT/final_nominal32.json" --num-envs 32 --steps 100 --nominal > "$VALIDATION_ROOT/final_nominal32.log" 2>&1
"$PYTHON_BIN" tools/probe_training_collision_runtime.py --source-root "$PROJECT_ROOT" --asset "$ASSET_PATH" --output "$VALIDATION_ROOT/final_randomized32.json" --num-envs 32 --steps 500 > "$VALIDATION_ROOT/final_randomized32.log" 2>&1
