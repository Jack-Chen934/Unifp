#!/usr/bin/env bash
# 2026-11-08用户已授权正式seed1启动，累计12000轮先做效果评估。
set -euo pipefail
cd /mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/source_launch12000_20261108/UniFP_v3
export CUDA_VISIBLE_DEVICES=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TMPDIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/b_tmp
export MPLCONFIGDIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/b_mpl
export TORCH_EXTENSIONS_DIR=/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions
export PYTHONPATH=.:/mnt/data_8THDD/myl_unifp/isaacgym/python
export LD_LIBRARY_PATH=/mnt/data_8THDD/conda_envs/unifp_v3_stage1/lib
sha256sum --check /mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/bt08_evidence_launch12000_20261108/tool_hashes.sha256
exec /mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python tools/train_v3_b_observed.py \
  --reproduction_part B --run_purpose formal --seed 1 --num_envs 4096 \
  --transition_from_a \
  --resume /mnt/data_8THDD/myl_unifp_v3_stage1/runs/a8000_seed1_v3_v6r2_resume0767_20261105/checkpoints/model_008000.pt \
  --parent_sha256 323a6688fe42559db1d0c545b778b456551ce0ac472f8169fec5ebce2d849dbc \
  --start_evidence /mnt/data_8THDD/myl_unifp_v3_stage1/validation/b_stage_20261107_r01/bt08_evidence_launch12000_20261108/start_evidence.json \
  --manifest configs/v3_b_initial_v1.json \
  --output_dir /mnt/data_8THDD/myl_unifp_v3_stage1/runs/b32000_seed1_v3_v6r2_from_a8000_20261108 \
  --sim_device cuda:0 --rl_device cuda:0 --pipeline gpu --num_threads 4 --headless
