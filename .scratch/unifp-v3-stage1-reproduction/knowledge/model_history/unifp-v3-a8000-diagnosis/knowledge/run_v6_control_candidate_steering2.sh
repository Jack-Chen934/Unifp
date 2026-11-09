#!/usr/bin/env bash
set -euo pipefail
DIAG_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/a8000_20261104/v6_r2_gpu
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r02/source/UniFP_v3
for motion_sign in 1 -1; do
    run_name="ground_factor2_gain170_steering2_sign${motion_sign}"
    bash "$DIAG_ROOT/run_candidate_r02.sh" "$DIAG_ROOT/probe_wheels.py" \
        --source-root "$PROJECT_ROOT" \
        --asset "$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf" \
        --gain-scale 1.7 --steering-gain-scale 2 --physics-step-factor 2 --motion-sign "$motion_sign" \
        --output "$DIAG_ROOT/$run_name.json" > "$DIAG_ROOT/$run_name.log" 2>&1
done
