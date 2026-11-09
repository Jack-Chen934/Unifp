"""核验训练器驻留GPU时，独立视频评估不改动其学习状态和RNG。"""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import isaacgym  # noqa: F401
import torch
from tools.check_v3_b_training import ObservedRunner, assert_tensor_tree_equal
from legged_gym.scripts import train_v3


class ResidencyRunner(ObservedRunner):
    """Retain the real training allocation while launching short graphics probes."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Learn normally, then verify isolated video evaluation alongside the learner.

        Args:
            num_learning_iterations: Explicit one-update engineering budget.
            init_at_random_ep_len: Unchanged production flag, required false.

        Raises:
            AssertionError: Parent RNG, model or either Adam changes during evaluation.
            subprocess.CalledProcessError: A native child cannot run beside the learner.

        Note:
            Only the diagnostic children's policy loops are shortened to two
            steps. Full-quota performance baselines are validated independently.
        """
        super().learn(num_learning_iterations, init_at_random_ep_len)
        checkpoint = self.log_dir / 'checkpoints' / f'model_{self.completed_iterations:06d}.pt'
        rng_before = self._capture_rng()
        model_before = {name: value.detach().clone() for name, value in self.alg.actor_critic.state_dict().items()}
        main_before = torch.load(checkpoint, map_location='cpu', weights_only=False)['main_optimizer_state_dict']
        auxiliary_before = torch.load(checkpoint, map_location='cpu', weights_only=False)['auxiliary_optimizer_state_dict']
        source = Path(train_v3.__file__).resolve().parents[2]
        reports = []
        for name in ('R2', 'contact', 'zero_control'):
            contact = name != 'R2'
            module = 'evaluate_v3_contact' if contact else 'evaluate_v3'
            program = (f'from legged_gym.scripts import {module} as entry; '
                       'entry.range=lambda stop: range(2 if stop >= 1000 else stop); entry.main()')
            output = self.log_dir / ('observation_probe_' + name)
            command = [sys.executable, '-c', program, '--reproduction_part', 'B', '--run_purpose', 'evaluation',
                '--seed', '1001', '--num_envs', '32', '--checkpoint', str(checkpoint),
                '--protocol', 'fixed_wall_contact' if contact else 'force_tracking', '--scenario', name,
                '--manifest', str(source / 'configs' / ('v3_b_contact_v1.json' if contact else 'v3_b_initial_v1.json')),
                '--output_dir', str(output), '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                '--pipeline', 'gpu', '--num_threads', '4', '--headless', '--video', '--smoke']
            with (self.log_dir / (name + '.console.log')).open('x') as console:
                subprocess.run(command, cwd=source, stdout=console, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / 'summary.json').read_text())
            assert summary['policy_updates'] == 0
            assert_tensor_tree_equal(self._capture_rng(), rng_before)
            assert_tensor_tree_equal(self.alg.actor_critic.state_dict(), model_before)
            assert_tensor_tree_equal(self.alg.optimizer.state_dict(), main_before)
            assert_tensor_tree_equal(self.alg.adaptation_module_optimizer.state_dict(), auxiliary_before)
            reports.append({'scenario': name, 'policy_updates': 0, 'parent_state_unchanged': True})
        report = {'passed': True, 'num_envs_training': 4096, 'num_envs_evaluation': 32,
            'parent_and_children_share_gpu': True, 'scenarios': reports,
            '说明': '父真实更新一轮后驻留；子视频评估每组仅两策略步，验证图形／显存／状态隔离，不替代正式配额基准。'}
        (self.log_dir / 'observation_residency_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    train_v3.V3OnPolicyRunner = ResidencyRunner
    train_v3.main()
