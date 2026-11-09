#!/usr/bin/env bash
set -euo pipefail
DIAG_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/diagnostics/a8000_20261104/v6_r2_gpu
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/a8000_repairs_20261104_r02/source/UniFP_v3
for physics_factor in 1 2; do
    gain_scale=1
    if [ "$physics_factor" -eq 2 ]; then gain_scale=1.7; fi
    run_name="timing32_factor${physics_factor}"
    bash "$DIAG_ROOT/run_candidate_r02.sh" "$DIAG_ROOT/probe_control_timing.py" \
        --source-root "$PROJECT_ROOT" \
        --asset "$PROJECT_ROOT/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/urdf/a1arm_rgm3_desktop_fixed_gripper.urdf" \
        --physics-step-factor "$physics_factor" --gain-scale "$gain_scale" \
        --output "$DIAG_ROOT/$run_name.json" > "$DIAG_ROOT/$run_name.log" 2>&1
done
