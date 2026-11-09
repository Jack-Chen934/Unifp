"""固定真实观测，测量正负命令力与等效几何目标的策略响应。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


def group_rms(values: torch.Tensor) -> list[float]:
    """Measure action changes separately for steering, drive and arm groups.

    Args:
        values: Dimensionless action changes [N,14] in policy order.

    Returns:
        Three RMS values over samples and each action group; no physical units.
    """
    return [float(values[:, start:end].square().mean().sqrt()) for start, end in ((0, 4), (4, 8), (8, 14))]


def shift_input(observations: torch.Tensor, axis: int, force_delta_n: float, geometric: bool) -> torch.Tensor:
    """Perturb only the latest real observation frame at the agreed command seam.

    Args:
        observations: Real normalized Actor histories [N,1856].
        axis: Yaw-frame XYZ coordinate index, 0 through 2.
        force_delta_n: Signed command perturbation in N.
        geometric: If true, shift geometric goal by delta/200 m instead of force.

    Returns:
        Independent histories with only current command force or spherical goal
        changed; old history and all measured physical channels remain identical.

    Note:
        These are local counterfactual inputs, not physically replayed histories.
        Force and equivalent goal tasks need not share future trajectories.
    """
    changed = observations.clone()
    current = changed[:, -58:]
    if not geometric:
        current[:, 52 + axis] += force_delta_n * .01
        return changed
    scales = current.new_tensor((.5, 1., 1.3))
    radius, elevation, azimuth = (current[:, 46:49] / scales).unbind(-1)
    cartesian = torch.stack((radius * elevation.cos() * azimuth.cos(),
        radius * elevation.cos() * azimuth.sin(), radius * elevation.sin()), -1)
    cartesian[:, axis] += force_delta_n / 200.
    new_radius = cartesian.norm(dim=-1).clamp_min(1e-6)
    sphere = torch.stack((new_radius, torch.asin((cartesian[:, 2] / new_radius).clamp(-1., 1.)),
        torch.atan2(cartesian[:, 1], cartesian[:, 0])), -1)
    current[:, 46:49] = sphere * scales
    return changed


def policy_parts(model: torch.nn.Module, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate original shared Encoder and Actor without Gaussian sampling.

    Args:
        model: Frozen V3ActorCritic in evaluation mode on CPU.
        observations: Normalized real or counterfactual inputs [N,1856].

    Returns:
        Latent [N,64] and raw action mean [N,14], without clipping or execution.
    """
    latent = model.adaptation_encoder_module(observations)
    mean = model.actor_body(torch.cat((observations[:, -58:], latent), -1))
    return latent, mean


def main() -> None:
    """Audit three frozen policies on seven identical real-input datasets.

    Raises:
        AssertionError: Input shapes, intervention isolation, decomposition, or
            actual policy distribution parity fail.
        OSError: Frozen source, checkpoints or real observation inputs are absent.

    Note:
        CPU inference only. No simulation, optimizer, RNG sampling or weight writes.
        Response magnitude and cosine do not establish correct control direction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--probes', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    module_path = args.source / 'legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py'
    specification = importlib.util.spec_from_file_location('command_policy', module_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    models, hashes = {}, {}
    for iteration in (9000, 10000, 12000):
        checkpoint = args.checkpoints / f'model_{iteration:06d}.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = module.V3ActorCritic(1856, 294, 12, 58, 14, **payload['config_snapshot']['training']['policy'])
        model.load_state_dict(payload['model_state_dict'], strict=True)
        models[iteration] = model.eval()
        hashes[str(iteration)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    records, datasets = [], []
    with torch.inference_mode():
        for probe in sorted(args.probes.glob('probe_*')):
            observation_path = probe / 'shared_observations.pt'
            if not observation_path.exists():
                continue
            observations = torch.load(observation_path, map_location='cpu', weights_only=False)
            assert observations.shape == (160, 1856) and torch.isfinite(observations).all()
            datasets.append({'probe': probe.name, 'sha256': hashlib.sha256(observation_path.read_bytes()).hexdigest(),
                'samples': len(observations), 'force_current_min_n': (observations[:, -58:][:, 52:55].min(0).values * 100).tolist(),
                'force_current_max_n': (observations[:, -58:][:, 52:55].max(0).values * 100).tolist()})
            for iteration, model in models.items():
                latent, baseline = policy_parts(model, observations)
                model.update_distribution(observations)
                torch.testing.assert_close(model.action_mean, baseline, rtol=0., atol=0.)
                for delta in (1., 10.):
                    for axis in range(3):
                        current_force = observations[:, -58 + 52 + axis] * 100.
                        valid = current_force.abs() + delta <= 60.
                        full_history = observations[:, :58].abs().sum(-1) > 0
                        positive = shift_input(observations, axis, delta, False)
                        negative = shift_input(observations, axis, -delta, False)
                        torch.testing.assert_close(positive[:, :-58], observations[:, :-58], rtol=0., atol=0.)
                        changed_indices = (positive - observations).abs().sum(0).nonzero().flatten().tolist()
                        assert changed_indices == [1856 - 58 + 52 + axis]
                        positive_latent, positive_mean = policy_parts(model, positive)
                        _, negative_mean = policy_parts(model, negative)
                        _, positive_goal_mean = policy_parts(model, shift_input(observations, axis, delta, True))
                        _, negative_goal_mean = policy_parts(model, shift_input(observations, axis, -delta, True))
                        # Exact fixed-order split: total = current-frame direct path
                        # at old latent + Encoder path at the new current frame.
                        direct = model.actor_body(torch.cat((positive[:, -58:], latent), -1)) - baseline
                        indirect = positive_mean - model.actor_body(torch.cat((positive[:, -58:], latent), -1))
                        torch.testing.assert_close(positive_mean - baseline, direct + indirect, rtol=1e-4, atol=1e-5)
                        force_response = (positive_mean - negative_mean) / (2. * delta)
                        goal_response = (positive_goal_mean - negative_goal_mean) / (2. * delta)
                        for subset, mask in (('all_valid', valid), ('full_history_valid', valid & full_history)):
                            if not mask.any():
                                continue
                            response = force_response[mask]
                            reference = goal_response[mask]
                            record: dict[str, Any] = {'probe': probe.name, 'iteration': iteration,
                                'axis': 'XYZ'[axis], 'delta_n': delta, 'subset': subset, 'samples': int(mask.sum()),
                                'force_derivative_group_rms_per_n': group_rms(response),
                                'equivalent_goal_derivative_group_rms_per_n': group_rms(reference),
                                'force_goal_cosine_by_group': [float(torch.nn.functional.cosine_similarity(response[:, start:end], reference[:, start:end], dim=-1).mean())
                                    for start, end in ((0, 4), (4, 8), (8, 14))],
                                'positive_delta_total_group_rms': group_rms((positive_mean - baseline)[mask]),
                                'positive_delta_direct_group_rms': group_rms(direct[mask]),
                                'positive_delta_encoder_group_rms': group_rms(indirect[mask]),
                                'latent_delta_rms': float((positive_latent - latent)[mask].square().mean().sqrt()),
                                'force_derivative_by_action_rms_per_n': response.square().mean(0).sqrt().tolist()}
                            records.append(record)
    result = {'checkpoint_sha256': hashes, 'source_policy_sha256': hashlib.sha256(module_path.read_bytes()).hexdigest(),
        'datasets': datasets, 'records': records,
        '说明': '当前帧正负局部扰动，旧历史与物理输入固定；动作敏感性和几何参考cos非控制正确性。RMS路径不能相加分责。'}
    output = Path(__file__).resolve().parent / 'policy_response.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'datasets': len(datasets), 'records': len(records), 'checkpoint_sha256': hashes}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
