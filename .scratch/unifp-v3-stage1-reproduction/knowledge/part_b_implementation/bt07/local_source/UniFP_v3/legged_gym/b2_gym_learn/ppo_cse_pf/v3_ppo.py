"""Official joint PPO mathematics with confirmed V3 terminal and auxiliary fixes."""

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as functional

from .v3_actor_critic import V3ActorCritic
from .v3_rollout_storage import V3RolloutStorage


class V3PPO:
    """Optimize one joint policy/value objective followed by one auxiliary objective."""

    def __init__(self, actor_critic: V3ActorCritic, num_learning_epochs: int = 5,
                 num_mini_batches: int = 4, clip_param: float = 0.2, gamma: float = 0.99,
                 lam: float = 0.95, value_loss_coef: float = 1.0, entropy_coef: float = 0.01,
                 learning_rate: float = 1e-3, max_grad_norm: float = 1.0,
                 use_clipped_value_loss: bool = True, schedule: str = "adaptive",
                 desired_kl: float = 0.01, device: str = "cpu") -> None:
        """Build the two Adam optimizers using the effective official configuration.

        Args:
            actor_critic: Fixed-schema V3 network shared by both optimizers.
            num_learning_epochs: Passes over each rollout, confirmed baseline 5.
            num_mini_batches: Equal batches per pass, confirmed baseline 4.
            clip_param: Joint PPO ratio and value-difference clipping radius.
            gamma: Reward discount per 20 ms policy step.
            lam: GAE trace decay per policy step.
            value_loss_coef: Main objective's value-loss multiplier.
            entropy_coef: Joint action entropy bonus multiplier.
            learning_rate: Initial main Adam rate, before KL adaptation.
            max_grad_norm: Main parameter global L2 gradient limit.
            use_clipped_value_loss: Whether to use the official clipped value loss.
            schedule: Main learning-rate mode, adaptive or fixed.
            desired_kl: Joint old-to-new KL target for adaptive scheduling.
            device: Explicit model/storage device.

        Raises:
            ValueError: Epoch/minibatch counts or learning-rate mode are invalid.
        """
        if type(num_learning_epochs) is not int or num_learning_epochs < 1 or type(num_mini_batches) is not int or num_mini_batches < 1 or schedule not in {"adaptive", "fixed"}:
            raise ValueError("Invalid V3 PPO update configuration")
        self.device = torch.device(device)
        self.actor_critic = actor_critic.to(self.device)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.learning_rate, self.max_grad_norm = learning_rate, max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule, self.desired_kl = schedule, desired_kl
        named_parameters = list(actor_critic.named_parameters())
        self.main_parameter_names = [name for name, _ in named_parameters if not name.startswith("adaptation_decoder_module.")]
        self.auxiliary_parameter_names = [name for name, _ in named_parameters if name.startswith(("adaptation_encoder_module.", "adaptation_decoder_module."))]
        parameter_map = dict(named_parameters)
        self.main_parameters = [parameter_map[name] for name in self.main_parameter_names]
        self.auxiliary_parameters = [parameter_map[name] for name in self.auxiliary_parameter_names]
        self.optimizer = torch.optim.Adam(self.main_parameters, lr=learning_rate)
        self.adaptation_module_optimizer = torch.optim.Adam(self.auxiliary_parameters, lr=1e-5)
        self.transition = V3RolloutStorage.Transition()
        self.update_in_progress = False
        self.storage: V3RolloutStorage

    def init_storage(self, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: Sequence[int], critic_obs_shape: Sequence[int],
                     obs_pred_shape: Sequence[int], action_shape: Sequence[int]) -> None:
        """Validate sampling dimensions before allocating the confirmed rollout.

        Args:
            num_envs: Parallel environment count N.
            num_transitions_per_env: Policy steps T, required to be 24.
            actor_obs_shape: Required [1856] flattened Actor history.
            critic_obs_shape: Required [294] flattened Critic history.
            obs_pred_shape: Required [12] environment supervision vector.
            action_shape: Required [14] raw sampled joint action.

        Raises:
            ValueError: Schema, rollout length or minibatch divisibility is invalid.
        """
        if num_transitions_per_env != 24 or tuple(map(tuple, (actor_obs_shape, critic_obs_shape, obs_pred_shape, action_shape))) != ((1856,), (294,), (12,), (14,)):
            raise ValueError("V3 rollout schema mismatch")
        if num_envs * 24 % self.num_mini_batches:
            raise ValueError("Minibatches must divide all rollout samples")
        self.storage = V3RolloutStorage(num_envs, 24, actor_obs_shape, critic_obs_shape, obs_pred_shape, action_shape, str(self.device))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor, obs_pred: torch.Tensor) -> torch.Tensor:
        """Capture detached old-policy quantities at one immutable action boundary.

        Args:
            obs: Published Actor input [N,1856], unchanged by subsequent env.step.
            critic_obs: Same-boundary privileged input [N,294].
            obs_pred: Same-boundary environment labels [N,12], never predictions.

        Returns:
            Raw Gaussian samples [N,14], before environment execution clipping.

        Raises:
            RuntimeError: A prior action is still waiting to be stored.
        """
        if self.transition.actions is not None:
            raise RuntimeError("Previous action has not been paired with an environment step")
        model = self.actor_critic
        self.transition.actions = model.act(obs).detach()
        self.transition.values = model.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = model.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = model.action_mean.detach()
        self.transition.action_sigma = model.action_std.detach()
        self.transition.observations = obs.detach()
        self.transition.critic_observations = critic_obs.detach()
        self.transition.observation_preds = obs_pred.detach()
        return self.transition.actions

    @torch.no_grad()
    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: Dict[str, Any]) -> None:
        """Bootstrap pure timeouts from their true terminal successor exactly once.

        Args:
            rewards: Environment rewards [N]; cloned before training compensation.
            dones: Integer or boolean episode-end mask [N].
            infos: Explicit failure/timeout masks, unordered done IDs, matching
                terminal Critic rows [K,294], and completed episode records.

        Raises:
            ValueError: Terminal masks, IDs or snapshot rows are inconsistent.
            RuntimeError: No sampled action is waiting for its successor.

        Note:
            Sets each episode record's PPO-only bonus; never changes raw rewards.
        """
        if self.transition.actions is None:
            raise RuntimeError("An environment step requires a preceding sampled action")
        count = self.storage.num_envs
        terminated, timeouts = infos["terminated"], infos["time_outs"]
        for mask in (terminated, timeouts):
            if mask.shape != (count,) or mask.dtype != torch.bool or mask.device != self.device:
                raise ValueError("Invalid terminal mask")
        if dones.shape != (count,) or dones.device != self.device or dones.dtype not in (torch.bool, torch.int64, torch.int32) or not ((dones == 0) | (dones == 1)).all():
            raise ValueError("Invalid environment dones")
        if torch.any(terminated & timeouts) or not torch.equal(dones.bool(), terminated | timeouts):
            raise ValueError("Failure and pure timeout masks must partition done environments")
        terminal_ids, terminal_obs = infos["terminal_env_ids"], infos["terminal_privileged_obs"]
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if terminal_ids.dtype != torch.long or terminal_ids.device != self.device or terminal_ids.ndim != 1 or not torch.equal(terminal_ids.sort().values, done_ids):
            raise ValueError("Terminal IDs must match all done rows exactly once")
        if terminal_obs.shape != (terminal_ids.numel(), 294) or terminal_obs.device != self.device or terminal_obs.dtype != torch.float32 or not torch.isfinite(terminal_obs).all():
            raise ValueError("Invalid terminal Critic snapshots")
        bonuses = torch.zeros(count, device=self.device)
        timeout_rows = timeouts[terminal_ids]
        if timeout_rows.any():
            bonuses[terminal_ids[timeout_rows]] = self.gamma * self.actor_critic.evaluate(terminal_obs[timeout_rows]).squeeze(-1)
        self.transition.rewards = rewards.clone() + bonuses
        self.transition.dones = dones.detach()
        self.storage.add_transitions(self.transition)
        for record in infos["episode_records"]:
            record["timeout_bootstrap_bonus"] = float(bonuses[record["env_id"]])
        self.transition.clear()
        self.actor_critic.reset(dones)

    @torch.no_grad()
    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Evaluate only surviving successors and build one done-masked GAE.

        Args:
            last_critic_obs: Published successor inputs [N,294] after step 24.

        Raises:
            RuntimeError: Collection is incomplete.
        """
        if self.storage.step != 24:
            raise RuntimeError("Returns require all 24 policy steps")
        last_values = torch.zeros(self.storage.num_envs, 1, device=self.device)
        surviving = ~self.storage.dones[-1, :, 0]
        if surviving.any():
            last_values[surviving] = self.actor_critic.evaluate(last_critic_obs[surviving])
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    @staticmethod
    def _check_gradients(parameters: List[torch.nn.Parameter]) -> None:
        """Reject non-finite gradients before changing optimizer state.

        Args:
            parameters: Ordered parameters owned by the current optimizer.

        Raises:
            FloatingPointError: Any present gradient is non-finite.
        """
        valid = [torch.isfinite(parameter.grad).all() for parameter in parameters if parameter.grad is not None]
        if valid and not torch.stack(valid).all():
            raise FloatingPointError("V3 optimizer gradients are non-finite")

    def update(self) -> Tuple[float, float, float, Dict[str, float]]:
        """Apply the official joint PPO step then a freshly recomputed auxiliary step.

        Returns:
            Mean value loss, surrogate loss, auxiliary loss and four group means,
            each divided by its actual update count, with no extra division by 64.

        Raises:
            RuntimeError: Storage is incomplete or returns are stale.
            FloatingPointError: KL, losses, gradients or learned parameters are invalid.

        Note:
            Failure leaves update_in_progress set, preventing boundary checkpoints.
            Clears the rollout only after every main/auxiliary update succeeds.
        """
        if not self.storage.returns_ready or self.storage.step != 24:
            raise RuntimeError("PPO update requires a full rollout and fresh returns")
        self.update_in_progress = True
        model = self.actor_critic
        sums = [0.0, 0.0, 0.0]
        auxiliary_sums = {label: 0.0 for label in model.adaptation_labels}
        updates = 0
        for obs, critic_obs, labels, actions, old_values, advantages, returns, old_logp, old_mu, old_sigma, masks in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            # Keep the official extra Gaussian sample: losses still use stored actions.
            model.zero_grad(set_to_none=True)
            model.act(obs, masks=masks)
            logp = model.get_actions_log_prob(actions)
            value = model.evaluate(critic_obs)
            if self.schedule == "adaptive":
                with torch.no_grad():
                    kl = (torch.log(model.action_std / old_sigma + 1e-5)
                          + (old_sigma.square() + (old_mu - model.action_mean).square())
                          / (2.0 * model.action_std.square()) - 0.5).sum(-1).mean()
                    if not torch.isfinite(kl):
                        raise FloatingPointError("V3 PPO KL is non-finite")
                    if kl > 2.0 * self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            ratio = (logp - old_logp.squeeze(-1)).exp()
            advantage = advantages.squeeze(-1)
            surrogate = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).mean()
            value_errors = (value - returns).square()
            if self.use_clipped_value_loss:
                clipped_value = old_values + (value - old_values).clamp(-self.clip_param, self.clip_param)
                value_errors = torch.maximum(value_errors, (clipped_value - returns).square())
            value_loss = value_errors.mean()
            main_loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * model.entropy.mean()
            if not torch.isfinite(main_loss):
                raise FloatingPointError("V3 main loss is non-finite")
            main_loss.backward()
            self._check_gradients(self.main_parameters)
            torch.nn.utils.clip_grad_norm_(self.main_parameters, self.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()

            # The shared encoder has changed: recompute its prediction after the main step.
            model.zero_grad(set_to_none=True)
            prediction = model.get_student_latent(obs)
            group_losses = [functional.mse_loss(prediction[:, 3 * index:3 * index + 3] * weight,
                                               labels[:, 3 * index:3 * index + 3] * weight)
                            for index, weight in enumerate(model.adaptation_weights)]
            auxiliary_loss = sum(group_losses)
            if not torch.isfinite(auxiliary_loss):
                raise FloatingPointError("V3 auxiliary loss is non-finite")
            auxiliary_loss.backward()
            self._check_gradients(self.auxiliary_parameters)
            self.adaptation_module_optimizer.step()
            model.zero_grad(set_to_none=True)
            for index, loss in enumerate((value_loss, surrogate, auxiliary_loss)):
                sums[index] += float(loss.detach())
            for label, loss in zip(model.adaptation_labels, group_losses):
                auxiliary_sums[label] += float(loss.detach())
            updates += 1
        if not torch.stack([torch.isfinite(parameter).all() for parameter in model.parameters()]).all() or not (model.std > 0.0).all():
            raise FloatingPointError("V3 updated model/std is invalid")
        self.storage.clear()
        self.update_in_progress = False
        return (*[value / updates for value in sums], {name: value / updates for name, value in auxiliary_sums.items()})
