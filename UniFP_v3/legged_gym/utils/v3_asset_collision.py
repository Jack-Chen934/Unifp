"""Apply only the desktop asset's evidenced assembly collision exceptions."""

import json
from pathlib import Path
from typing import Any, List


def apply_desktop_assembly_filters(gym: Any, env: Any, actor: int,
                                   manifest_path: Path) -> List[dict]:
    """Filter wheel mounting solids and the fixed closed-finger assembly.

    Args:
        gym: Isaac Gym API object.
        env: Environment containing the new desktop-based robot.
        actor: Actor handle in env.
        manifest_path: Asset manifest with source-group-to-shape indices.

    Returns:
        Applied named local rules with affected body-relative shape indices.

    Raises:
        ValueError: Imported shape counts do not match the generated asset.

    Note:
        Updates actor shape filter masks. Unrelated chassis solids, wheel hubs,
        and all nonadjacent arm/payload collision pairs keep their original masks.
    """
    manifest = json.loads(manifest_path.read_text())
    names = gym.get_actor_rigid_body_names(env, actor)
    spans = gym.get_actor_rigid_body_shape_indices(env, actor)
    properties = gym.get_actor_rigid_shape_properties(env, actor)
    rules = []
    for name, span in zip(names, spans):
        if span.count != manifest['collision_solid_counts'][name]['count']:
            raise ValueError(f'Imported shape count differs for {name}')
    for bit, corner in enumerate(('fl', 'fr', 'bl', 'br')):
        mount_indices = manifest['collision_solid_counts']['base_link']['source_groups'][corner + '_fixed_collision']
        wheel = corner + '_Link'
        wheel_count = spans[names.index(wheel)].count
        rules.append({'bodies': ['base_link', wheel], 'local_shapes': [mount_indices, list(range(wheel_count))],
                      'mask': 1 << bit, '依据': '本轮固定装配件与滚动件的局部CAD凸包接触；其他底盘形状不豁免。'})
    rules.append({'bodies': ['gripper1', 'gripper2'],
                  'local_shapes': [list(range(spans[names.index(name)].count)) for name in ('gripper1', 'gripper2')],
                  'mask': 1 << 4, '依据': '两指在固定闭合位姿的内部CAD交叠；对其他部件的碰撞保留。'})
    for rule in rules:
        for name, local_indices in zip(rule['bodies'], rule['local_shapes']):
            span = spans[names.index(name)]
            for local_index in local_indices:
                properties[span.start + local_index].filter |= rule['mask']
    clearance_records = sorted(manifest.get('visual_clearance_records', []),
                               key=lambda record: record['wheel'])
    rolling_records = {record['link']: record for record in manifest.get('rolling_collision_records', [])}
    for bit, record in enumerate(clearance_records, start=5):
        wheel = record['wheel'] + '_Link'
        tire_shapes = rolling_records[wheel]['shape_indices']
        mask = 1 << bit
        clearance_rule = {'bodies': ['base_link', wheel],
                          'local_shapes': [[record['local_shape']], tire_shapes],
                          'mask': mask,
                          '依据': '后轮胎面与对应base visual间的紧密装配间隙；仅豁免该低位碰撞与胎面分片。',
                          'source_component': record['source_component']}
        rules.append(clearance_rule)
        for name, local_indices in zip(clearance_rule['bodies'], clearance_rule['local_shapes']):
            span = spans[names.index(name)]
            for local_index in local_indices:
                properties[span.start + local_index].filter |= mask
    gym.set_actor_rigid_shape_properties(env, actor, properties)
    return rules
