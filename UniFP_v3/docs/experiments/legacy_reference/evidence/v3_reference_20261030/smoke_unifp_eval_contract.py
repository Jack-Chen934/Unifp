"""Real-simulation regression: terminal capture, reset-independent goals, force timing."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from isaaclab.app import AppLauncher
from unifp_eval_protocol import make_schedule, validate_state

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path('logs/evaluation-contract.json'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import torch
from isaaclab.utils.math import quat_from_euler_xyz
from unifp_vec_env import UniFPVecEnv
from unifp_force_curriculum import UniFPForceCurriculum


def main():
    env = UniFPVecEnv(num_envs=2, device=args.device, domain_randomization=False,
                      observation_noise=False, command_sampling=False, force_curriculum=False,
                      capture_diagnostics=True)
    try:
        # Exercise the actual termination implementation for both signs, inside/outside thresholds.
        for angle,expected in ((0.79,False),(-0.79,False),(0.81,True),(-0.81,True)):
            r=torch.full((2,),angle,device=env.device); z=torch.zeros_like(r)
            fake=SimpleNamespace(data=SimpleNamespace(root_quat_w=quat_from_euler_xyz(r,z,z)))
            done,_=env.rewards.termination(fake)
            assert done.tolist()==[expected]*2, ('roll',angle)
        for angle,expected in ((0.99,False),(-0.99,False),(1.01,True),(-1.01,True)):
            p=torch.full((2,),angle,device=env.device); z=torch.zeros_like(p)
            fake=SimpleNamespace(data=SimpleNamespace(root_quat_w=quat_from_euler_xyz(z,p,z)))
            done,_=env.rewards.termination(fake)
            assert done.tolist()==[expected]*2, ('pitch',angle)
        curriculum=UniFPForceCurriculum(2,env.device)
        for step in (0,192000):
            curriculum.update(step,env.episode_length_buf)
        assert curriculum.enabled_steps==0
        for substep in range(4):
            curriculum.update(192001,env.episode_length_buf,substep_index=substep)
        assert curriculum.enabled_steps==1 and curriculum.enabled_physics_updates==4
        schedule=make_schedule(30,123)
        env.goal.install_schedule(schedule)
        env.reset()
        timeouts=[]
        with torch.inference_mode():
            for step in range(20):
                state=env.state_snapshot()
                validate_state({k:v.tolist() for k,v in state.items()})
                expected=torch.tensor(schedule['samples'][step]['sphere'],device=env.device)
                assert torch.allclose(state['goal_sphere'],expected.expand(2,-1),atol=1e-6)
                # Deliberately give envs different reset histories at the same tape position.
                if step in (2,5):
                    env.episode_length_buf[0 if step==2 else 1]=1000
                _,reward,done,extras=env.step(torch.zeros(2,17,device=env.device))
                assert torch.equal(state['goal_sphere'],expected.expand(2,-1))
                terminal=extras['pre_reset_state']
                validate_state({k:v.tolist() for k,v in terminal.items()})
                assert torch.allclose(reward,torch.stack(list(extras['reward_terms'].values())).sum(0))
                if step in (2,5):
                    i=0 if step==2 else 1
                    assert done[i] and extras['termination_reasons']['timeout'][i]
                    assert terminal['episode_steps'][i]==1001 and env.episode_length_buf[i]==0
                    timeouts.append({'step':step+1,'env_id':i,'terminal_length':1001,'reset_length':0})
        report={'passed':True,'steps':20,'forced_timeouts':timeouts,
                'signed_tilt_thresholds_checked':True,'snapshots_immutable':True,
                'goals_independent_of_resets':True,'force_boundary':curriculum.summary()}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print('UNIFP_EVAL_CONTRACT_PASS',json.dumps(report),flush=True)
    finally:
        env.close()
        app.close(skip_cleanup=True)


if __name__=='__main__':
    main()
