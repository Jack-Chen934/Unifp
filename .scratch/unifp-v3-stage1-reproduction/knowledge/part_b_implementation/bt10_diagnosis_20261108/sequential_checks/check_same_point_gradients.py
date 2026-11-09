"""对保存的真实rollout在同一参数点核对PPO与辅助梯度方向。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def main() -> None:
    """Compare encoder gradient directions without stepping any optimizer.

    Raises:
        OSError: A checkpoint, rollout or policy module cannot be read.
        AssertionError: Stored old policy likelihood differs from the loaded model.

    Note:
        Uses CPU autograd on three disjoint rollout subsets per checkpoint. The
        critic loss and entropy have no encoder gradient in this architecture.
        Cosine conflict measures local directions, not closed-loop causality.
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    specification=importlib.util.spec_from_file_location('gradient_policy',args.source/'legged_gym/b2_gym_learn/ppo_cse_pf/v3_actor_critic.py')
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    results=[]
    for iteration in (9000,10000,12000):
        path=args.root/f'check_{iteration}'/'rollout.pt'
        if not path.exists():
            continue
        rollout=torch.load(path,map_location='cpu',weights_only=False)
        payload=torch.load(args.checkpoints/f'model_{iteration:06d}.pt',map_location='cpu',weights_only=False)
        model=module.V3ActorCritic(1856,294,12,58,14,**payload['config_snapshot']['training']['policy'])
        model.load_state_dict(payload['model_state_dict'],strict=True)
        parameters=list(model.adaptation_encoder_module.parameters())
        subsets=[]
        for subset in range(3):
            # Interleave disjoint samples across all 24 times and 128 actors.
            observations=rollout['observations'].flatten(0,1)[subset::3]
            actions=rollout['actions'].flatten(0,1)[subset::3]
            labels=rollout['observation_preds'].flatten(0,1)[subset::3]
            advantages=rollout['advantages'].flatten(0,1)[subset::3,0]
            old_logp=rollout['actions_log_prob'].flatten(0,1)[subset::3,0]
            model.update_distribution(observations)
            logp=model.get_actions_log_prob(actions)
            difference=float((logp-old_logp).abs().max())
            assert difference<1e-3
            ratio=(logp-old_logp).exp()
            clipped=ratio.clamp(1-payload['config_snapshot']['training']['algorithm']['clip_param'],1+payload['config_snapshot']['training']['algorithm']['clip_param'])
            policy_loss=torch.maximum(-advantages*ratio,-advantages*clipped).mean()
            prediction=model.get_student_latent(observations)
            auxiliary_loss=sum(((prediction[:,3*i:3*i+3]-labels[:,3*i:3*i+3])*weight).square().mean() for i,weight in enumerate(model.adaptation_weights))
            policy_gradient=torch.cat([value.flatten() for value in torch.autograd.grad(policy_loss,parameters)])
            auxiliary_gradient=torch.cat([value.flatten() for value in torch.autograd.grad(auxiliary_loss,parameters)])
            subsets.append({'samples':len(observations),'max_old_logp_difference':difference,
                'same_point_encoder_gradient_cosine':float(torch.nn.functional.cosine_similarity(policy_gradient,auxiliary_gradient,dim=0)),
                'policy_gradient_l2':float(policy_gradient.norm()),'auxiliary_gradient_l2':float(auxiliary_gradient.norm()),
                'auxiliary_loss':float(auxiliary_loss.detach())})
        results.append({'iteration':iteration,'subsets':subsets})
    (args.root/'same_point_gradients.json').write_text(json.dumps({'results':results,
        '说明':'同参数点、同批次真实观测监督；梯度cos负值表示局部方向冲突，非退化唯一根因。梯度范数不能忽略Adam历史而直接换算参数更新比。'},
        ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
