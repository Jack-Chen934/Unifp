"""独立实验身份差异的最小单元测试。"""

import copy
from pathlib import Path
import unittest

from experiment_contract import experiment_snapshot, validate_runtime_move


class ContractTests(unittest.TestCase):
    """Test only experiment-specific config and runtime seams."""

    def test_single_learning_change_and_no_mutation(self) -> None:
        """Keep all learning fields except entropy and the explicit purpose label."""
        parent = {'environment': {'run': {'run_purpose': 'formal', 'reproduction_part': 'B'},
                  'asset': {'file': '/old/UniFP_v3/resources/robots/v6/urdf/robot.urdf'}, 'env': {'num_envs': 4096}},
                  'training': {'runner': {'save_interval': 200, 'tensorboard': True},
                  'algorithm': {'entropy_coef': .01, 'learning_rate': .001}, 'seed': 1}}
        preserved = copy.deepcopy(parent)
        result = experiment_snapshot(parent, Path('/new/UniFP_v3'), 0.)
        self.assertEqual(parent, preserved)
        expected = copy.deepcopy(parent)
        expected['environment']['run']['run_purpose'] = 'development'
        expected['environment']['asset']['file'] = '/new/UniFP_v3/resources/robots/v6/urdf/robot.urdf'
        expected['training']['algorithm']['entropy_coef'] = 0.
        expected['training']['runner'].update(save_interval=50, tensorboard=False)
        self.assertEqual(result, expected)

    def test_reject_other_entropy_settings(self) -> None:
        """Reject settings outside the two user-approved experimental arms."""
        with self.assertRaises(ValueError):
            experiment_snapshot({}, Path('/source'), .005)

    def test_only_gpu_visibility_may_move(self) -> None:
        """Accept explicit physical GPU relocation with identical logical devices."""
        parent = {'cuda_visible_devices': '1', 'device': 'cuda:0', 'torch': 'same'}
        actual = {**parent, 'cuda_visible_devices': '3'}
        self.assertEqual(validate_runtime_move(parent, actual),
                         {'cuda_visible_devices': {'parent': '1', 'target': '3'}})

    def test_reject_dependency_or_logical_device_change(self) -> None:
        """Keep software versions and RNG's logical CUDA device strict."""
        parent = {'cuda_visible_devices': '1', 'device': 'cuda:0', 'torch': 'same'}
        for change in ({'device': 'cuda:1'}, {'torch': 'different'}):
            with self.assertRaises(ValueError):
                validate_runtime_move(parent, {**parent, **change})


if __name__ == '__main__':
    unittest.main()
