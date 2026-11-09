"""限定熵消融的配置与物理GPU迁移接口。"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def experiment_snapshot(parent: dict[str, Any], source_root: Path, entropy_coef: float) -> dict[str, Any]:
    """Build the approved experiment config without mutating its parent.

    Args:
        parent: Complete checkpoint config, with the same 4096-actor B task.
        source_root: Frozen checkout used to relocate the existing asset path.
        entropy_coef: Approved coefficient, exactly 0.01 for control or 0.0.

    Returns:
        Deep-copied config with explicit development purpose and experiment outputs.

    Raises:
        ValueError: The coefficient is unapproved or the asset has no portable suffix.
    """
    if entropy_coef not in (0., .01):
        raise ValueError('Only the approved entropy arms 0.01 and 0.0 are allowed')
    target = copy.deepcopy(parent)
    parts = Path(target['environment']['asset']['file']).parts
    target['environment']['asset']['file'] = str(source_root.joinpath(*parts[parts.index('resources'):]))
    target['environment']['run']['run_purpose'] = 'development'
    target['training']['algorithm']['entropy_coef'] = entropy_coef
    target['training']['runner'].update(save_interval=50, tensorboard=False)
    return target


def validate_runtime_move(parent: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Permit only a recorded physical GPU visibility change.

    Args:
        parent: Saved interpreter, dependencies, simulator and CUDA runtime fields.
        actual: Independently measured fields for the isolated experiment process.

    Returns:
        Explicit old/new values for the allowed visibility difference, possibly empty.

    Raises:
        ValueError: Software, logical device, threads or another runtime field differs.
    """
    changes = {key: {'parent': parent.get(key), 'target': actual.get(key)}
               for key in parent.keys() | actual.keys() if parent.get(key) != actual.get(key)}
    if set(changes) - {'cuda_visible_devices'}:
        raise ValueError('Unapproved runtime difference: ' + str(changes))
    return changes
