#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/mnt/data_8THDD/myl_unifp_v3_stage1/validation/collision_optimization_20261103_r02/source/UniFP_v3
export GIT_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/evaluation_provenance/a8000_collision_r02_20261104.git
export GIT_WORK_TREE="$PROJECT_ROOT"
cd "$PROJECT_ROOT"
# 元数据在训练结束后的评估期建立；不将其冒充训练前提交。
git init -b evaluation-snapshot "$GIT_DIR"
git config user.name sorry
git config user.email sorry_8168@qq.com
git add -A
git commit -m "Record frozen A8000 source snapshot at evaluation time"
git rev-parse HEAD
git status --short
