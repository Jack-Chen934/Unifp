"""离线分解共享表示与Actor变化，不创建仿真或优化器。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def rms_groups(values: torch.Tensor) -> list[float]:
    """Measure per-action RMS changes for the three policy joint groups.

    Args:
        values: Dimensionless action differences [samples,14].

    Returns:
        RMS over samples and actions for steering, drive and arm respectively.
    """
    return [float(values[:, start:end].square().mean().sqrt())
            for start, end in ((0, 4), (4, 8), (8, 14))]


def main() -> None:
    """Compare checkpoint modules on the same saved real observation histories.

    Raises:
        OSError: A source, observation or checkpoint file cannot be read.
        AssertionError: The action decomposition or source observation width differs.

    Note:
        Runs on CPU, writes JSON only, and never changes checkpoint tensors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    module_path = args.source / 'legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py'
    specification = importlib.util.spec_from_file_location('latent_policy', module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    models, checkpoint_hashes = {}, {}
    for iteration in (9000, 10000, 12000):
        path = args.checkpoints / f'model_{iteration:06d}.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        model = module.V3ActorCritic(1856, 294, 12, 58, 14,
                                    **payload['config_snapshot']['training']['policy'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        models[iteration] = model.eval()
        checkpoint_hashes[str(iteration)] = hashlib.sha256(path.read_bytes()).hexdigest()
    results: list[dict[str, Any]] = []
    for probe in sorted(args.probes.glob('probe_*')):
        observation_path = probe / 'shared_observations.pt'
        if not observation_path.exists():
            continue
        observations = torch.load(observation_path, map_location='cpu', weights_only=False)
        assert observations.ndim == 2 and observations.shape[1] == 1856
        with torch.inference_mode():
            latents = {key: model.adaptation_encoder_module(observations) for key, model in models.items()}
            means = {(actor, encoder): model.actor_body(torch.cat((observations[:, -58:], latent), -1))
                     for actor, model in models.items() for encoder, latent in latents.items()}
            entry: dict[str, Any] = {'probe': probe.name, 'samples': len(observations),
                                     'observation_sha256': hashlib.sha256(observation_path.read_bytes()).hexdigest(),
                                     'comparisons': []}
            for iteration in (10000, 12000):
                # This identity separates changes in a fixed order, not causality:
                # new_actor(new_encoder)-old_actor(old_encoder) = encoder + actor.
                total = means[iteration, iteration] - means[9000, 9000]
                encoder_effect = means[iteration, iteration] - means[iteration, 9000]
                actor_effect = means[iteration, 9000] - means[9000, 9000]
                assert torch.allclose(total, encoder_effect + actor_effect, atol=1e-5)
                latent_delta = latents[iteration] - latents[9000]
                entry['comparisons'].append({
                    'iteration': iteration, 'total_action_rms': rms_groups(total),
                    'encoder_change_with_new_actor_rms': rms_groups(encoder_effect),
                    'actor_change_with_old_encoder_rms': rms_groups(actor_effect),
                    'encoder_change_with_old_actor_rms': rms_groups(means[9000, iteration] - means[9000, 9000]),
                    'latent_rms_change': float(latent_delta.square().mean().sqrt()),
                    'latent_rms_reference': float(latents[9000].square().mean().sqrt()),
                    'latent_cosine_mean': float(torch.nn.functional.cosine_similarity(latents[iteration], latents[9000]).mean()),
                })
        results.append(entry)
    args.output.write_text(json.dumps({'checkpoint_sha256': checkpoint_hashes, 'datasets': results,
        '说明': '同输入离线模块替换；RMS分项不是可加百分比，混合模块有分布不匹配，不能直接当闭环修复。'},
        ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'datasets': len(results), 'first': results[0]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
