#!/usr/bin/env bash
set -euo pipefail
DIAG_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/a8000_20261104/v6_r2_gpu
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r02/source/UniFP_v3
for physics_factor in 1 2; do
    run_name="airborne_gain120_factor${physics_factor}"
    bash "$DIAG_ROOT/run_candidate_r02.sh" "$DIAG_ROOT/probe_wheels.py" \
        --source-root "$PROJECT_ROOT" \
        --asset "$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf" \
        --base-height 1000 --gain-scale 1.2 --physics-step-factor "$physics_factor" \
        --output "$DIAG_ROOT/$run_name.json" > "$DIAG_ROOT/$run_name.log" 2>&1
done
