"""Exercise A08 rewards and A09 state contracts in real Preview4 environments."""

import json
from pathlib import Path
import sys
from typing import Any, Dict

from isaacgym import gymapi
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg


def check_runtime() -> Dict[str, Any]:
    """Run reward/randomization/reset/push checks without constructing a learner.

    Returns:
        Chinese-keyed runtime evidence suitable for the local experiment record.

    Raises:
        AssertionError: A confirmed physical, reward or restoration contract fails.

    Note:
        Creates eight Gym actors, advances physics, and destroys the simulator.
        No network or optimizer is constructed and no training is started.
    """
    torch.manual_seed(31)
    cfg = V3PosForceCfg()
    cfg.env.num_envs = 8
    cfg.noise.add_noise = False
    cfg.run.reproduction_part = "A"
    cfg.run.run_purpose = "development"
    params = gymapi.SimParams()
    params.dt = cfg.sim.dt
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = True
    params.physx.num_position_iterations = 4
    params.physx.num_velocity_iterations = 1
    params.physx.contact_offset = cfg.sim.physx.contact_offset
    params.physx.rest_offset = 0.0
    environment = V3PosForce(cfg, params, gymapi.SIM_PHYSX, "cuda:0", True)
    try:
        names = environment.gym.get_actor_rigid_body_names(environment.envs[0], environment.actor_handles[0])
        for env_id in range(environment.num_envs):
            properties = environment.gym.get_actor_rigid_body_properties(
                environment.envs[env_id], environment.actor_handles[env_id]
            )
            base = properties[names.index("base_link")]
            delta = float(environment.mass_com_delta[env_id, 0])
            assert abs(base.mass - (65.8456 + delta)) < 1e-4
            assert abs(base.inertia.x.x / base.mass - 1.6878422133333333 / 65.8456) < 1e-6
            assert 0.3 <= float(environment.friction[env_id]) <= 2.0
        old_scale = environment.execution_scale_delta.clone()
        environment.reset_idx(torch.tensor([1], device=environment.device))
        torch.testing.assert_close(environment.execution_scale_delta[2:], old_scale[2:])
        assert not torch.equal(environment.execution_scale_delta[1], old_scale[1])

        # Keep one untrained action sequence to verify finite rewards and real episode samples.
        actions = torch.zeros(8, 14, device=environment.device)
        reward_min, reward_max = float("inf"), float("-inf")
        physical_failures = []
        for step_index in range(24):
            _, reward, done, info = environment.step(actions)
            for record in info["episode_records"]:
                assert record["end_type"] == "physical_failure"
                assert record["steps"] > 0
                assert record["timeout_bootstrap_bonus"] is None
                physical_failures.append({"策略步": step_index + 1, "记录": record})
            assert torch.isfinite(reward).all()
            assert len(environment.reward_names) == 23
            reward_min = min(reward_min, float(reward.min()))
            reward_max = max(reward_max, float(reward.max()))
        state = environment.get_training_state()
        saved_friction = environment.friction.clone()
        expected_random_sequence = torch.rand(5)
        # Creation physics stays fixed after prepare_sim; cross-process creation
        # restoration is exercised by ticket05's GPU 512-environment resume run.
        environment.reset_idx(torch.tensor([7], device=environment.device))
        environment.load_training_state(state)
        torch.testing.assert_close(environment.friction, saved_friction)
        torch.testing.assert_close(torch.rand(5), expected_random_sequence)
        restored_shapes = environment.gym.get_actor_rigid_shape_properties(
            environment.envs[0], environment.actor_handles[0]
        )
        assert all(abs(shape.friction - float(saved_friction[0])) < 1e-5 for shape in restored_shapes)
        torch.testing.assert_close(environment.execution_scale_delta, state["execution_scale_delta"].to(environment.device))

        # Publishing progress must not inject motion; actual push occurs once in step.
        previous_root = environment.root_states.clone()
        environment.set_training_progress(16, 400)
        torch.testing.assert_close(environment.root_states, previous_root)
        _, _, done, info = environment.step(actions)
        assert info["base_push_applied"]
        assert environment.last_push_policy_step == 400
        assert torch.count_nonzero(environment.external_ee_force_y) == 0
        _, _, _, repeated_info = environment.step(actions)
        assert not repeated_info["base_push_applied"]

        # Force a pure timeout decision without adding hundreds of unnecessary steps.
        expected_episode_steps = int(environment.episode_metric_steps[0]) + 1
        other_episode_steps = int(environment.episode_metric_steps[1]) + 1
        environment.episode_length_buf[0] = environment.max_episode_length - 1
        _, _, done, info = environment.step(actions)
        assert bool(done[0]) and bool(info["time_outs"][0])
        record = next(record for record in info["episode_records"] if record["env_id"] == 0)
        assert record["steps"] == expected_episode_steps
        assert record["end_type"] == "timeout"
        assert record["duration_s"] == expected_episode_steps * environment.dt
        assert len(record["raw_reward_means"]) == 23
        assert environment.episode_metric_steps[0] == 0
        assert environment.episode_metric_steps[1] == other_episode_steps
        return {
            "环境数": 8, "策略步": 27, "奖励项数": 23,
            "奖励范围": [reward_min, reward_max], "创建物性与Critic回读一致": True,
            "质量变化时惯量按同比例缩放": True, "reset执行系数局部采样": True,
            "训练时钟setter不推进物理": True, "push单次触发且末端力标签为零": True,
            "创建物性只读核验与可变状态恢复": True, "完成episode记录": record,
            "未训练零动作物理失败记录": physical_failures,
        }
    finally:
        environment.gym.destroy_sim(environment.sim)


if __name__ == "__main__":
    report = check_runtime()
    output = Path(__file__).resolve().parents[2] / ".scratch/unifp-v3-stage1-reproduction/knowledge/a08_a09_runtime_20261031.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
