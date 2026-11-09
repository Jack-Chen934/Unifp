"""Portable evaluation inputs and validation; does not import Isaac Sim."""
from __future__ import annotations
import hashlib
import json
import math
import random
from pathlib import Path

POLICY_DT = 0.02
REWARD_ATOL = 2e-5


def schedule_hash(schedule):
    data = {key: value for key, value in schedule.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _valid_segment(start, goal):
    for i in range(10):
        r, p, y = [a + (b-a)*i/9 for a, b in zip(start, goal)]
        x, yy, z = r*math.cos(p)*math.cos(y), r*math.cos(p)*math.sin(y), r*math.sin(p)
        if z < -0.7 or (-0.7 < x < 0.25 and -0.2 < yy < 0.2 and -0.8 < z < -0.15):
            return False
    return True


def make_schedule(steps, seed):
    """One immutable yaw-frame command tape shared by all evaluation envs.

    Uses released ranges/interpolation and ten collision samples. Unlike training,
    rejection must succeed (no ten-attempt fallback); resets never restart the tape.
    First segment matches the released initial-to-first-goal motion.
    """
    rng = random.Random(seed)
    start, goal = [0.66, math.pi/4, 0.0], [0.66, 0.0, 0.0]
    rows, segment = [], 0
    orientation = [0.0]*3
    while len(rows) <= steps:
        duration = rng.uniform(1.0, 3.0)/POLICY_DT
        total = duration + rng.uniform(0.5, 2.0)/POLICY_DT
        for timer in range(math.floor(total)+1):
            t = min(timer/duration, 1.0)
            rows.append({'sphere': [a+(b-a)*t for a,b in zip(start,goal)],
                         'orientation_delta': orientation.copy(), 'segment': segment})
            if len(rows) > steps:
                break
        start = goal
        for _ in range(10000):
            goal = [rng.uniform(0.35,0.95), rng.uniform(-2*math.pi/5,2*math.pi/5),
                    rng.uniform(-3*math.pi/5,3*math.pi/5)]
            if _valid_segment(start,goal):
                break
        else:
            raise RuntimeError('Unable to generate collision-free evaluation segment')
        orientation = [rng.uniform(-0.5,0.5) for _ in range(3)]
        segment += 1
    schedule = {'schema_version': 1, 'seed': seed, 'policy_dt': POLICY_DT,
                'frame': 'base yaw; center = root XY + yaw_rotate(0.2,0,0.8)',
                'reset_semantics': 'global tape continues across automatic resets', 'samples': rows}
    schedule['sha256'] = schedule_hash(schedule)
    return schedule


def load_schedule(path, steps):
    schedule = json.loads(Path(path).read_text())
    if schedule.get('schema_version') != 1 or schedule.get('policy_dt') != POLICY_DT:
        raise ValueError('Unsupported goal schedule schema or timestep')
    if schedule_hash(schedule) != schedule.get('sha256'):
        raise ValueError('Goal schedule hash mismatch')
    if len(schedule['samples']) <= steps:
        raise ValueError('Goal schedule needs steps+1 samples')
    for row in schedule['samples']:
        if any(len(row[k]) != 3 or not all(math.isfinite(x) for x in row[k])
               for k in ('sphere','orientation_delta')):
            raise ValueError('Invalid goal schedule sample')
    return schedule


def validate_state(state, atol=2e-6):
    for p,g,e,n in zip(state['ee_position_world_m'],state['goal_position_world_m'],
                       state['position_error_xyz_m'],state['position_error_m']):
        if max(abs(a-b-c) for a,b,c in zip(p,g,e)) > atol:
            raise AssertionError('Saved ee-goal differs from saved xyz error')
        if abs(math.sqrt(sum(x*x for x in e))-n) > atol:
            raise AssertionError('Saved xyz error norm differs from scalar error')


def validate_trace(payload):
    """Checks mathematical identities and the pre-action/pre-reset time contract."""
    if payload.get('schema_version', 0) >= 2:
        done_count = sum(sum(row['done']) for row in payload['samples'])
        if done_count != len(payload['episodes']) or done_count != payload['report']['done_count']:
            raise AssertionError('Episode records or summary disagree with done flags')
    for row in payload['samples']:
        validate_state(row)
        if payload.get('schema_version',0) >= 2:
            if abs(row['policy_time_s'] - (row['step']-1)*POLICY_DT) > 1e-7:
                raise AssertionError('Incorrect pre-action timestamp')
            transition = row['transition']
            state = transition['pre_reset_state']
            validate_state(state)
            if abs(transition['policy_time_s']-row['step']*POLICY_DT)>1e-7:
                raise AssertionError('Incorrect transition timestamp')
            masks = transition['termination_reasons']
            for i,done in enumerate(row['done']):
                if bool(done) != any(values[i] for values in masks.values()):
                    raise AssertionError('Done differs from termination reason union')
                if masks['roll'][i] != (abs(state['roll_rad'][i])>0.8):
                    raise AssertionError('Roll termination threshold mismatch')
                if masks['pitch'][i] != (abs(state['pitch_rad'][i])>1.0):
                    raise AssertionError('Pitch termination threshold mismatch')
                if masks['timeout'][i] != (state['episode_steps'][i]>1000):
                    raise AssertionError('Timeout threshold mismatch')
                if abs(sum(v[i] for v in transition['reward_terms'].values())-row['reward'][i])>REWARD_ATOL:
                    raise AssertionError('Weighted reward terms do not sum to total reward')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('trace',type=Path)
    args = parser.parse_args()
    validate_trace(json.loads(args.trace.read_text()))
    print('UNIFP_TRACE_VALIDATED', args.trace)
