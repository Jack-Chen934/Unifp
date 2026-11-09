def gym_ranger_backend(num_envs, seed, max_force_n=15., measure_dof_forces=False, fixed_com=False,
                       init_before_prepare=True, official_shared=False):
    """Gym transport for existing Ranger task code, extracted without Kit imports.

    Only simulator I/O is implemented here. Observation, PD target equations,
    reset, task step, reward, goal, randomization and force code execute from
    the production sources, without rewriting their formulas.
    """
    from isaacgym import gymapi, gymtorch
    import torch
    import numpy as np
    import random
    import math
    import ast
    import __future__
    from types import SimpleNamespace as NS
    torch.set_num_threads(8)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    ns = dict(torch=torch,np=np,random=random,math=math,Path=Path,
        copysign=lambda mag,sign:torch.copysign(torch.full_like(sign,mag),sign))
    def extract(relative, names):
        path = ROOT/relative
        tree = ast.parse(path.read_text())
        selected = []
        for node in tree.body:
            if getattr(node,'name',None) in names:
                node.decorator_list = []
                selected.append(node)
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                if any(isinstance(t,ast.Name) and t.id in names for t in targets): selected.append(node)
        found = {getattr(n,'name',None) for n in selected}
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec',
            flags=__future__.annotations.compiler_flag),ns)
        assert all(name in ns for name in names), (relative,names,found)
    extract('IsaacLab/source/isaaclab/isaaclab/utils/math.py',
        ['normalize','euler_xyz_from_quat','quat_from_euler_xyz','yaw_quat',
         'quat_apply','quat_apply_inverse','quat_apply_yaw'])
    extract('unifp_observations.py',['COMMAND_SCALES','UniFPActorInputs'])
    extract('unifp_domain_randomization.py',['UniFPDomainRandomization'])
    extract('unifp_force_curriculum.py',['OFFICIAL_START_POLICY_STEP','POLICY_DT','_ForcePulse','UniFPForceCurriculum'])
    extract('unifp_task_signals.py',['field_slices','sphere_to_cart','cart_to_sphere','UniFPEEGoal'])
    extract('unifp_task_rewards.py',['REWARD_SCALES','POLICY_DT','MAX_EPISODE_STEPS','UniFPTaskRewards'])
    ns.update(ARM_NAMES=['arm_joint'+str(i) for i in range(1,7)],
        ARM_EFFORT_LIMITS=dict(zip(['arm_joint'+str(i) for i in range(1,7)],[40.,40.,27.,7.,7.,7.])),
        PHYSICS_DT=.005,DECIMATION=4,ASSET_VERSION='v3_original',USD_PATH=Path('unused'),RangerSceneCfg=None)
    extract('ranger_unifp_vec_env.py',['ROOT_HEIGHT','CORNERS','ACTION_NAMES','ARM_HOME',
        'CRITIC_FIELDS','CRITIC_SLICES','CRITIC_SIZE','RangerJointPD','RangerPrivilegedInputs','RangerRewards'])
    ns['ACTION_JOINT_NAMES']=ns['ACTION_NAMES']
    extract('unifp_vec_env.py',['UniFPVecEnv'])
    extract('ranger_unifp_vec_env.py',['RangerUniFPVecEnv'])
    gym = gymapi.acquire_gym()
    lab_root=OUT/'gym_ranger_20261024/lab_transport_defaults'
    lab_metadata=json.loads((lab_root/'result.json').read_text())
    lab_properties=torch.load(lab_root/'shape_properties.pt',map_location='cpu')
    schema_path=Path('/var/local/sorry/conda/envs/isaaclab-unifp/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/PhysxSchema/resources/generatedSchema.usda')
    import re
    defaults={key:(value if value is not None else float(re.search(
        r'float physxRigidBody:'+key[0].lower()+key[1:]+r' = ([0-9.]+)',schema_path.read_text()).group(1)))
        for key,value in lab_metadata['rigid_body_schema_defaults'].items()}
    params = gymapi.SimParams()
    params.dt=.005; params.substeps=1; params.up_axis=gymapi.UP_AXIS_Z
    params.gravity=gymapi.Vec3(0,0,-9.81); params.use_gpu_pipeline=False
    params.physx.use_gpu=True; params.physx.solver_type=1
    params.physx.num_position_iterations=32; params.physx.num_velocity_iterations=1
    params.physx.rest_offset=0.
    params.physx.bounce_threshold_velocity=.5
    params.physx.friction_offset_threshold=.04
    params.physx.friction_correlation_distance=.025
    params.physx.max_depenetration_velocity=defaults['MaxDepenetrationVelocity']
    sim = gym.create_sim(0,-1,gymapi.SIM_PHYSX,params)
    assert sim
    plane=gymapi.PlaneParams(); plane.normal=gymapi.Vec3(0,0,1)
    plane.static_friction=plane.dynamic_friction=1.
    gym.add_ground(sim,plane)
    opts=gymapi.AssetOptions()
    opts.collapse_fixed_joints=True; opts.replace_cylinder_with_capsule=False
    opts.default_dof_drive_mode=gymapi.DOF_MODE_NONE; opts.use_mesh_materials=False
    opts.linear_damping=defaults['LinearDamping']; opts.angular_damping=defaults['AngularDamping']
    asset_root=ROOT/'robot_description/a1arm_description/urdf'
    asset=gym.load_asset(sim,str(asset_root),'a1arm_rgm3_original_movable.urdf',opts)
    assert asset
    joint_names=gym.get_asset_dof_names(asset); body_names=gym.get_asset_rigid_body_names(asset)
    props=gym.get_asset_dof_properties(asset)
    shape_spans=gym.get_asset_rigid_body_shape_indices(asset)
    span_by_body=dict(zip(body_names,shape_spans))
    # Both importers retain the same collision XML and merge the same fixed
    # links. Map flattened Lab offsets by named body, then collision ordinal.
    lab_offsets={};cursor=0
    for name in lab_metadata['body_names']:
        count=int(span_by_body[name].count)
        lab_offsets[name]=lab_properties['contact_offsets'][0,cursor:cursor+count]
        cursor+=count
    assert cursor==lab_properties['contact_offsets'].numel()
    q0=torch.zeros(len(joint_names))
    for name,value in zip(ns['ARM_NAMES'],ns['ARM_HOME']): q0[joint_names.index(name)]=value
    kp=torch.full_like(q0,20.); kd=torch.full_like(q0,2.); limits=torch.full_like(q0,10.)
    for i,name in enumerate(joint_names):
        mode=gymapi.DOF_MODE_POS
        if 'steering' in name: kp[i]=300.; kd[i]=20.; limits[i]=35.
        elif 'drive' in name: kp[i]=0.; kd[i]=20.; limits[i]=80.; mode=gymapi.DOF_MODE_VEL
        elif name in ns['ARM_NAMES']: kp[i]=80.; kd[i]=4.; limits[i]=ns['ARM_EFFORT_LIMITS'][name]
        props['driveMode'][i]=mode
    props['stiffness']=kp.numpy(); props['damping']=kd.numpy(); props['effort']=limits.numpy()
    envs=[]; actors=[]
    for i in range(num_envs):
        env=gym.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),int(math.ceil(math.sqrt(num_envs))))
        pose=gymapi.Transform(); pose.p=gymapi.Vec3(0,0,.32)
        actor=gym.create_actor(env,asset,pose,'ranger',i,1)
        if measure_dof_forces: gym.enable_actor_dof_force_sensors(env,actor)
        shape_props=gym.get_actor_rigid_shape_properties(env,actor)
        for name,span in span_by_body.items():
            for ordinal in range(span.count):
                shape_props[span.start+ordinal].contact_offset=float(lab_offsets[name][ordinal])
                shape_props[span.start+ordinal].rest_offset=0.
                shape_props[span.start+ordinal].friction=1.
        gym.set_actor_rigid_shape_properties(env,actor,shape_props)
        gym.set_actor_dof_properties(env,actor,props)
        state=np.zeros(len(joint_names),dtype=gymapi.DofState.dtype); state['pos']=q0.numpy()
        gym.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL)
        gym.set_actor_dof_position_targets(env,actor,q0.numpy())
        gym.set_actor_dof_velocity_targets(env,actor,np.zeros(len(joint_names),np.float32))
        envs.append(env);actors.append(actor)
    sim_prepared=False
    class BodyView:
        # Implements just the init-time Lab domain randomizer property API.
        def __init__(self):
            self.ps=[gym.get_actor_rigid_body_properties(e,a) for e,a in zip(envs,actors)]
            self.ms=[gym.get_actor_rigid_shape_properties(e,a) for e,a in zip(envs,actors)]
        def read_bodies(self):
            self.ps=[gym.get_actor_rigid_body_properties(e,a) for e,a in zip(envs,actors)]
            return self.ps
        def get_masses(self): return torch.tensor([[p.mass for p in ps] for ps in self.read_bodies()])
        def get_coms(self): return torch.tensor([[[p.com.x,p.com.y,p.com.z,0.,0.,0.,1.] for p in ps] for ps in self.read_bodies()])
        def get_inertias(self): return torch.tensor([[[getattr(getattr(p.inertia,r),c) for r in 'xyz' for c in 'xyz'] for p in ps] for ps in self.read_bodies()])
        def get_material_properties(self):
            self.ms=[gym.get_actor_rigid_shape_properties(e,a) for e,a in zip(envs,actors)]
            return torch.tensor([[[p.friction,p.friction,p.restitution] for p in ps] for ps in self.ms])
        def set_masses(self,values,ids):
            assert not (init_before_prepare and sim_prepared), 'Rigid properties must be initialized before prepare_sim'
            for i in ids.tolist():
                for j,p in enumerate(self.ps[i]):p.mass=float(values[i,j])
                gym.set_actor_rigid_body_properties(envs[i],actors[i],self.ps[i],False)
        def set_coms(self,values,ids):
            assert not (init_before_prepare and sim_prepared), 'COM must be initialized before prepare_sim'
            # Keep the production randomizer and every RNG draw. Intervene at
            # simulator I/O before it reads back the realized privileged COM.
            if fixed_com:
                values=values.clone()
                values[:,body_names.index('base_link'),:3]=baseline_com[:,body_names.index('base_link'),:3]
            for i in ids.tolist():
                for j,p in enumerate(self.ps[i]):p.com=gymapi.Vec3(*values[i,j,:3].tolist())
                gym.set_actor_rigid_body_properties(envs[i],actors[i],self.ps[i],False)
        def set_inertias(self,values,ids):
            assert not (init_before_prepare and sim_prepared), 'Inertia must be initialized before prepare_sim'
            for i in ids.tolist():
                for j,p in enumerate(self.ps[i]):
                    m=values[i,j].view(3,3)
                    inertia=gymapi.Mat33()
                    inertia.x=gymapi.Vec3(*m[0].tolist())
                    inertia.y=gymapi.Vec3(*m[1].tolist())
                    inertia.z=gymapi.Vec3(*m[2].tolist())
                    p.inertia=inertia
                gym.set_actor_rigid_body_properties(envs[i],actors[i],self.ps[i],False)
        def set_material_properties(self,values,ids):
            assert not (init_before_prepare and sim_prepared), 'Materials must be initialized before prepare_sim'
            for i in ids.tolist():
                for j,p in enumerate(self.ms[i]): p.friction=float(values[i,j,0]);p.restitution=float(values[i,j,2])
                gym.set_actor_rigid_shape_properties(envs[i],actors[i],self.ms[i])
    body_view=BodyView()
    # Preserve the original baseline before applying any randomized property.
    baseline_com=body_view.get_coms()
    baseline_inertia=body_view.get_inertias()
    baseline_mass=body_view.get_masses()
    staged_domain=None
    if init_before_prepare:
        # Execute the existing randomizer once, at its required Gym boundary.
        # Neither RangerJointPD nor UniFPActorInputs draws RNG before the
        # production domain constructor. Its only controller dependency here
        # is the number of actions; all subsequent goal/reset RNG is retained.
        staged_domain=ns['UniFPDomainRandomization'](
            NS(root_physx_view=body_view,body_names=body_names),
            NS(action_indices=torch.arange(len(ns['ACTION_NAMES']))),
            NS(num_envs=num_envs,device='cpu'),enabled=True,
            com_body='base_link',gripper_body='arm_seg6')
    gym.prepare_sim(sim)
    sim_prepared=True
    roots=gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof=gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim)).view(num_envs,len(joint_names),2)
    rb=gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim)).view(num_envs,len(body_names),13)
    contacts=gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim)).view(num_envs,len(body_names),3)
    sensed_effort=(gymtorch.wrap_tensor(gym.acquire_dof_force_tensor(sim)).view(num_envs,len(joint_names))
        if measure_dof_forces else None)
    forces=torch.zeros(num_envs,len(body_names),3); moments=torch.zeros_like(forces)
    pos_target=q0.expand(num_envs,-1).clone(); vel_target=torch.zeros_like(pos_target)
    class Robot:
        def __init__(self):
            self.joint_names=joint_names;self.body_names=body_names
            default=torch.zeros(num_envs,13);default[:,2]=.32;default[:,3]=1.
            self.data=NS(joint_pos=dof[:,:,0],joint_vel=dof[:,:,1],
                default_joint_pos=q0.expand(num_envs,-1).clone(),default_joint_vel=torch.zeros_like(pos_target),
                default_root_state=default,joint_pos_limits=torch.tensor(np.stack((props['lower'],props['upper']),-1)).unsqueeze(0),
                applied_torque=torch.zeros_like(pos_target))
            self.root_physx_view=body_view
        def refresh(self):
            gym.refresh_dof_state_tensor(sim);gym.refresh_actor_root_state_tensor(sim)
            gym.refresh_rigid_body_state_tensor(sim);gym.refresh_net_contact_force_tensor(sim)
            if measure_dof_forces:
                gym.refresh_dof_force_tensor(sim)
                self.data.measured_dof_force=sensed_effort
            d=self.data;d.root_pos_w=roots[:,:3];d.root_quat_w=roots[:,[6,3,4,5]]
            d.root_vel_w=roots[:,7:13];d.body_pos_w=rb[:,:,:3]
            d.body_quat_w=rb[:,:,[6,3,4,5]];d.body_vel_w=rb[:,:,7:13]
            d.root_lin_vel_b=ns['quat_apply_inverse'](d.root_quat_w,roots[:,7:10])
            d.root_ang_vel_b=ns['quat_apply_inverse'](d.root_quat_w,roots[:,10:13])
            d.projected_gravity_b=ns['quat_apply_inverse'](d.root_quat_w,torch.tensor([0.,0.,-1.]).expand(num_envs,-1))
        def set_joint_position_target(self,target):pos_target.copy_(target)
        def set_joint_velocity_target(self,target):
            vel_target.copy_(target)
            self.data.applied_torque=torch.clamp(kp*(pos_target-dof[:,:,0])+kd*(vel_target-dof[:,:,1]),-limits,limits)
        def set_joint_effort_target(self,target):assert torch.count_nonzero(target)==0
        def set_external_force_and_torque(self,*,forces,torques,body_ids,is_global):
            assert is_global
            # This API replaces the prior wrench each physics substep.
            external_forces.zero_();external_forces[:,body_ids]=forces
            moments.zero_();moments[:,body_ids]=torques
        def write_root_pose_to_sim(self,pose,env_ids):
            roots[env_ids,:3]=pose[:,:3];roots[env_ids,3:7]=pose[:,[4,5,6,3]]
            ids=env_ids.to(torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(roots),gymtorch.unwrap_tensor(ids),len(ids))
            self.refresh()
        def write_root_velocity_to_sim(self,velocity,env_ids=None):
            if env_ids is None:env_ids=torch.arange(num_envs)
            roots[env_ids,7:13]=velocity;ids=env_ids.to(torch.int32)
            gym.set_actor_root_state_tensor_indexed(sim,gymtorch.unwrap_tensor(roots),gymtorch.unwrap_tensor(ids),len(ids))
            self.refresh()
        def write_joint_state_to_sim(self,q,dq,env_ids):
            dof[env_ids,:,0]=q;dof[env_ids,:,1]=dq;ids=env_ids.to(torch.int32)
            gym.set_dof_state_tensor_indexed(sim,gymtorch.unwrap_tensor(dof),gymtorch.unwrap_tensor(ids),len(ids))
            self.refresh()
    external_forces=forces
    robot=Robot();robot.refresh()
    lab_indices=[lab_metadata['body_names'].index(name) for name in body_names]
    baseline_differences=dict(mass_kg=float((baseline_mass-lab_properties['masses'][:,lab_indices]).abs().max()),
        com_xyz_m=float((baseline_com[:,:,:3]-lab_properties['coms'][:,lab_indices,:3]).abs().max()),
        inertia_kg_m2=float((baseline_inertia-lab_properties['inertias'][:,lab_indices]).abs().max()))
    assert max(baseline_differences.values())<2e-6,baseline_differences
    class Scene:
        def __init__(self):
            self.num_envs=num_envs;self.device='cpu'
            # Gym API environment placement is global, but CPU tensor root and
            # rigid body states are environment-local. Keep all task vectors
            # in that same local world frame, including reset positions.
            self.gym_grid_origins=torch.tensor([[gym.get_env_origin(e).x,gym.get_env_origin(e).y,
                gym.get_env_origin(e).z] for e in envs])
            assert roots[:,:2].abs().max()<.01
            self.env_origins=torch.zeros(num_envs,3)
            self.items={'feet':NS(body_names=body_names,data=NS(net_forces_w=contacts)),
                        'robot_contacts':NS(body_names=body_names,data=NS(net_forces_w=contacts))}
        def __getitem__(self,key):return self.items[key]
        def reset(self,ids):
            external_forces[ids]=0.;moments[ids]=0.
            pos_target[ids]=robot.data.joint_pos[ids];vel_target[ids]=0.
        def write_data_to_sim(self):
            gym.set_dof_position_target_tensor(sim,gymtorch.unwrap_tensor(pos_target))
            gym.set_dof_velocity_target_tensor(sim,gymtorch.unwrap_tensor(vel_target))
            gym.apply_rigid_body_force_tensors(sim,gymtorch.unwrap_tensor(external_forces),gymtorch.unwrap_tensor(moments),gymapi.GLOBAL_SPACE)
        def update(self,dt):robot.refresh()
    class Simulation:
        def step(self):gym.simulate(sim);gym.fetch_results(sim,True)
        def clear_instance(self):gym.destroy_sim(sim)
    class RangerGym(ns['UniFPVecEnv']):
        num_single_obs=58;num_obs=1856;num_privileged_obs=294;num_pred_obs=12;num_actions=14
        critic_slices=ns['CRITIC_SLICES'];_create_task=ns['RangerUniFPVecEnv']._create_task
        def __init__(self):
            self.sim=Simulation();self.scene=Scene();self.robot=robot
            self.device='cpu';self.num_envs=num_envs;self._all_ids=torch.arange(num_envs)
            self.implicit_pd=True;self.capture_diagnostics=True
            if staged_domain is None:
                self._create_task(True,True,192001)
            else:
                constructor=ns['UniFPDomainRandomization']
                def use_staged(robot_arg,controller_arg,scene_arg,**kw):
                    assert robot_arg is robot and scene_arg is self.scene
                    assert len(controller_arg.action_indices)==staged_domain.num_motors
                    assert kw==dict(enabled=True,com_body='base_link',gripper_body='arm_seg6')
                    return staged_domain
                ns['UniFPDomainRandomization']=use_staged
                try:self._create_task(True,True,192001)
                finally:ns['UniFPDomainRandomization']=constructor
            # Set both independent pulse streams before the first reset. No
            # extra RNG draw, observation call, or reward/learner change.
            self.curriculum.command.max_force_n=float(max_force_n)
            self.curriculum.external.max_force_n=float(max_force_n)
            self.force_curriculum_enabled=False;self.safety_height_termination=False;self.rough_terrain=False
            self.push_robots_enabled=True;self.push_interval_steps=400;self.max_push_vel_xy=.8
            self.push_count=0;self.push_peak_velocity=0.
            self.observation_noise=True;self.command_sampling=True
            self.zero_wrench=torch.zeros(num_envs,1,3);self.gripper_id=self.privileged.ee_id
            self.global_policy_steps=0;self.episode_length_buf=self.rewards.episode_steps
            self.obs_buf=torch.zeros(num_envs,1856);self.privileged_obs_buf=torch.zeros(num_envs,294)
            self.obs_pred=torch.zeros(num_envs,12);self.rew_buf=torch.zeros(num_envs)
            self.reset_buf=torch.zeros(num_envs,dtype=torch.bool);self.extras={}
            self.transport=dict(physics_dt=.005,decimation=4,solver_iterations=[32,1],
                pipeline='GPU PhysX / CPU tensors and learner',self_collision=False,
                collapse_fixed_joints=True,replace_cylinder_with_capsule=False,
                rigid_body_defaults=defaults,baseline_mass_com_inertia_max_difference=baseline_differences,
                lab_property_evidence=str(lab_root),contact_offset_mapping='named body plus within-body collision ordinal',
                body_names=body_names,joint_names=joint_names,dof_properties={k:props[k].tolist() for k in props.dtype.names},
                fixed_com=fixed_com,rigid_property_initialization='before_prepare' if init_before_prepare else 'legacy_after_prepare',
                com_intervention='nominal base COM only; retain all randomizer draws' if fixed_com else 'original randomization',
                limitations=['Gym and Lab PhysX versions/collision cooking are different',
                    'Within-body merged collision ordinal was assumed from the same URDF, not geometrically certified',
                    'Gym drive torque reward uses the same clamped PD estimate as Lab implicit actuator, not measured force',
                    'CPU pipeline uses environment-local world coordinates'])
            self.diagnostic_gym_handles=NS(gym=gym,sim=sim,envs=envs,actors=actors)
            if official_shared:
                assert max_force_n==60. and init_before_prepare
                install_ranger_official_shared(self,ns,float(params.dt)*4)
                self.transport['official_shared']=self.shared_contract
            self.reset()
    return RangerGym(),ns
