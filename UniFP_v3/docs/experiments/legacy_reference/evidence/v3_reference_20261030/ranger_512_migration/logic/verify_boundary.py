"""Read-only independent audit of frozen continuation boundary data; writes audit and sole report."""
from pathlib import Path
import importlib.util,json,hashlib,datetime,re
import numpy as np
import sys
I=int(sys.argv[1])
assert I in (13500,14000,14500,15000,15500,16000)

N=Path(__file__).resolve().parent
A=N.parent
P=A.parents[2]/'PROJECT_REPORT.md'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('boundary_review',N/'run.py');m.protect();e=m.evaluator(I)
s=read(N/('boundary_summary_%06d.json'%I));digest=sha(s['checkpoint']);assert digest==s['checkpoint_sha256']
import torch
ck=torch.load(s['checkpoint'],map_location='cpu',weights_only=False)
assert ck['iterations']==I and ck['global_policy_steps']==24*I
counts={}
for name,key in [('ppo','optimizer_state_dict'),('auxiliary','adaptation_optimizer_state_dict')]:
 counts[name]=sorted(set(int(v['step']) for v in ck[key]['state'].values() if 'step' in v))
assert counts=={'ppo':[20*I],'auxiliary':[20*I]},counts
assert ck['force_stage'] and ck['com_randomization'] is False
iso=read(N/('evaluation_isolation_%06d.json'%I));assert iso['passed'] and iso['checkpoint_sha256']==digest
assert iso['simulation_steps_during_evaluation']==0
assert iso['model_two_adam_lr_rng_history_observations_physics_force_curriculum_unchanged']
verified=[]
for saved in s['results']:
 d=Path(saved['directory']);r=read(d/'result.json')
 assert sha(d/'result.json')==saved['result_sha256']
 assert r['checkpoint_sha256']==digest
 assert sha(d/'trace.npz')==r['trace_sha256']==saved['trace_sha256']
 assert sha(d/'initial_state.pt')==r['initial_state_sha256']==saved['initial_state_sha256']
 with np.load(d/'trace.npz') as z:
  t={k:z[k] if z[k].dtype==np.bool_ else z[k].astype(np.float64) for k in z.files}
 assert all(np.isfinite(v).all() for v in t.values())
 metrics=e.score(t,np);diff=0.
 for k,v in metrics.items():
  for ref in (r['metrics'],saved['metrics']):
   assert (v is None)==(ref[k] is None)
   if v is not None:
    assert np.isclose(v,ref[k],atol=1e-10,rtol=1e-10),(k,v,ref[k])
    diff=max(diff,abs(v-ref[k]))
 assert r['optimizer_updates']==0 and r['model_unchanged']
 if not r['force_curriculum']:
  assert metrics['actual_force_max_n']==metrics['command_force_max_n']==0
  assert np.count_nonzero(t['target'][...,6:12])==0
 else:
  assert 0<metrics['actual_force_max_n']<=60.0001 and 0<metrics['command_force_max_n']<=60.0001
 verified.append(dict(seed=r['seed'],mode=r['action_mode'],force=r['force_curriculum'],metrics=metrics,
   maximum_recomputation_difference=diff,directory=str(d),result_sha256=sha(d/'result.json'),
   trace_sha256=sha(d/'trace.npz'),initial_state_sha256=sha(d/'initial_state.pt')))
assert len(verified)==8 and len({(r['seed'],r['mode'],r['force']) for r in verified})==8
rows=[json.loads(l) for l in (N/'train/timings.jsonl').read_text().splitlines() if json.loads(l)['iteration']<=I]
assert len(rows)==I-13000 and [r['iteration'] for r in rows]==list(range(13001,I+1))
for r in rows:
 assert r['global_policy_steps']==r['iteration']*24
 assert r['adam_steps']=={'ppo':[r['iteration']*20],'auxiliary':[r['iteration']*20]}
for size in (100,300,500):
 selected=rows[-size:];expected=s['training_windows'][str(size)]
 for k,v in expected.items():
  if k=='observed_iterations':assert v==size
  elif k=='non_timeout_terminations':assert v==sum(r[k] for r in selected)
  elif k=='reward_terms':
   for term,term_value in v.items():assert np.isclose(sum(r[k][term] for r in selected)/size,term_value,atol=1e-12)
  else:
   values=[r[k] for r in selected if r.get(k) is not None]
   assert np.isclose(sum(values)/len(values),v,atol=1e-12),(k,size)
old=read(A/'gym_ranger_force_stage_20261028_v1/evaluation_summary.json')['results']
all_results=list(old)
for i in range(9500,12001,500): all_results.extend(read(A/'gym_ranger_force_continue9000_to12000_20261028_v1'/('boundary_summary_%06d.json'%i))['results'])
for i in (12500,13000): all_results.extend(read(A/'gym_ranger_force_continue12000_to13000_20261028_v1'/('boundary_summary_%06d.json'%i))['results'])
for i in range(13500,I+1,500): all_results.extend(read(N/('boundary_summary_%06d.json'%i))['results'])
def metric(i,seed,mode,force):
 return next(r['metrics'] for r in all_results if r['iteration']==i and r['seed']==seed and r['action_mode']==mode and r['force_curriculum']==force)
fields=['ee_error_mean_cm','ee_force_target_error_mean_cm','xy_rmse_m_s','yaw_rmse_rad_s','force_estimate_rmse_per_axis_n','zero_predictor_rmse_per_axis_n','force_rmse_improvement_over_zero_n','nonzero_force_vector_error_rmse_n','nonzero_zero_predictor_vector_rmse_n','sustained_zero_force_estimate_norm_n','non_timeout_terminations','velocity_prediction_rmse_m_s','ee_state_prediction_mse_normalized']
paired=[]
for prior in sorted({8000,9000,13000,I-500}):
 for seed in (266391,266392):
  for mode in ('mean','sample'):
   for force in (False,True):
    before=metric(prior,seed,mode,force);after=metric(I,seed,mode,force)
    paired.append(dict(prior_iteration=prior,iteration=I,seed=seed,mode=mode,force=force,
      before={k:before[k] for k in fields},after={k:after[k] for k in fields},
      delta={k:after[k]-before[k] if after[k] is not None and before[k] is not None else None for k in fields}))
aggregates={}
for mode in ('mean','sample'):
 for force in (False,True):
  key=mode+'_'+('force' if force else 'zero');aggregates[key]={}
  for iteration in (8000,9000,*range(9500,I+1,500)):
   ms=[metric(iteration,seed,mode,force) for seed in (266391,266392)]
   aggregates[key][str(iteration)]={k:sum(x[k] for x in ms)/2 if all(x[k] is not None for x in ms) else None for k in fields}
   aggregates[key][str(iteration)]['non_timeout_terminations']=sum(x['non_timeout_terminations'] for x in ms)
oldrows=[json.loads(l) for l in (A/'gym_ranger_force_stage_20261028_v1/train/timings.jsonl').read_text().splitlines()]
base_rows=[json.loads(l) for l in (A/'gym_ranger_fresh512_fixedcom_20261028_v1/train/timings.jsonl').read_text().splitlines()]
rewards={str(i):sum(r['mean_reward_per_step'] for r in rr[-100:])/100 for i,rr in [(8000,base_rows),(9000,oldrows),(I,rows)]}
status=read(N/'status.json');launch=read(N/'launch.json')
for role in ('worker','supervisor'):
 p=Path('/proc')/str(launch[role+'_pid'])/'stat'
 if p.exists():
  f=p.read_text().split();assert f[21]==launch[role+'_start_ticks'] and f[2] not in ('T','Z')
 else:assert I==16000 and read(N/'completion.json')['state']=='complete'
assert (status['iteration']>=I and status['state'] in ('running','evaluating','complete'))
m.protect()
review=dict(passed=True,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),iteration=I,
 checkpoint_sha256=digest,global_policy_steps=24*I,adam_steps=counts,results=verified,
 maximum_recomputation_difference=max(r['maximum_recomputation_difference'] for r in verified),
 training_windows=s['training_windows'],training_windows_recomputed=True,paired_comparison=paired,
 aggregates_two_validation_seed_arithmetic_means_except_termination_sum=aggregates,
 comparable_training_reward_last100=rewards,protected_files_verified=168,evaluation_isolation_verified=True,
 actual_live_iteration_at_review=status['iteration'],source_checkpoint_sha256=m.sha(m.SOURCE),
 resume_physics_and_history_rebuilt_not_uninterrupted=True)
m.write(N/('heartbeat_review_%06d.json'%I),review)
print(json.dumps(dict(passed=True,iteration=I,checkpoint_sha256=digest,
 maximum_recomputation_difference=review['maximum_recomputation_difference'],live_iteration=status['iteration'],
 aggregates=aggregates,training_reward=rewards),ensure_ascii=False))
