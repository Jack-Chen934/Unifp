
import os,sys,json,time,cProfile,pstats,io
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
os.environ['TORCH_EXTENSIONS_DIR']='/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions'
source='/mnt/data_8THDD/myl_unifp_v3_stage1/validation/ticket07_20261102_r03/UniFP_v3'
sys.path.insert(0,source)
sys.path.insert(0,'/mnt/data_8THDD/myl_unifp/isaacgym/python')
from isaacgym import gymapi,gymutil,gymtorch
import torch
import legged_gym.envs
from legged_gym.envs.v3.v3_pos_force_config import V3PosForceCfg
from legged_gym.envs.v3.legged_robot_v3 import V3PosForce
from legged_gym.utils.helpers import class_to_dict,set_seed
set_seed(1)
torch.cuda.set_device(0)
cfg=V3PosForceCfg()
cfg.env.num_envs=512
params=gymapi.SimParams()
gymutil.parse_sim_config(class_to_dict(cfg.sim),params)
params.use_gpu_pipeline=True
params.physx.use_gpu=True
created=time.perf_counter()
env=V3PosForce(cfg,params,gymapi.SIM_PHYSX,'cuda:0',True)
try:
    print('诊断环境已创建',time.perf_counter()-created,flush=True)
    actions=torch.zeros(512,14,device=env.device)
    for _ in range(4):
        env.step(actions)
        env.pending_episode_records=[]
    torch.cuda.synchronize()
    profiler=cProfile.Profile()
    started=time.perf_counter()
    profiler.enable()
    for _ in range(12):
        env.step(actions)
        env.pending_episode_records=[]
    torch.cuda.synchronize()
    profiler.disable()
    full_policy_step_seconds=(time.perf_counter()-started)/12
    profile_output=io.StringIO()
    pstats.Stats(profiler,stream=profile_output).sort_stats('cumtime').print_stats(30)
    torch.cuda.synchronize()
    started=time.perf_counter()
    for _ in range(48):
        torques=env._compute_v3_torques(actions)
        env.gym.set_dof_actuation_force_tensor(env.sim,gymtorch.unwrap_tensor(torques))
        env.gym.simulate(env.sim)
        env.gym.refresh_dof_state_tensor(env.sim)
    torch.cuda.synchronize()
    physics_48_substeps_seconds=time.perf_counter()-started
    result={'num_envs':512,'physical_gpu':1,'seed':1,'action_mode':'zero','full_policy_step_seconds':full_policy_step_seconds,'full_rollout_24_steps_seconds_estimate':full_policy_step_seconds*24,'physics_48_substeps_seconds':physics_48_substeps_seconds,'physics_96_substeps_seconds_estimate':physics_48_substeps_seconds*2,'actor_shape_count':len(env.gym.get_actor_rigid_shape_properties(env.envs[0],env.actor_handles[0])),'body_count':env.num_bodies,'profile':profile_output.getvalue(),'description':'独立诊断进程，无学习器、无PPO更新、无训练checkpoint，不改变GPU0正式训练；全步和纯物理段不具有完全相同的状态序列。'}
    print('PERFORMANCE_RESULT '+json.dumps(result,ensure_ascii=False),flush=True)
finally:
    env.gym.destroy_sim(env.sim)
