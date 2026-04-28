from typing import Callable

import numpy
import torch

from config import PPOConfig
from torch_ac.algos.base import BaseAlgo
import math

class PPO(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, model, device, config: PPOConfig, preprocess_obss: Callable, parallel=False,
                 aux_loss_coef: float = 0.0, transition_loss_coef: float = 0.0):
        """
        Args:
            aux_loss_coef: Coefficient for auxiliary chained distance prediction loss.
                          Set to 0.0 to disable auxiliary loss. Typical values: 0.01-0.1
            transition_loss_coef: Coefficient for transition prediction loss.
                          Set to 0.0 to disable. Typical values: 0.01-0.1
        """

        num_steps_per_proc = config.steps_per_process

        super().__init__(envs, model, device, num_steps_per_proc, config.discount, config.lr, config.gae_lambda,
                         config.entropy_coef, config.value_loss_coef, config.max_grad_norm, preprocess_obss, parallel=parallel)

        self.clip_eps = config.clip_eps
        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.act_shape = envs[0].action_space.shape
        self.aux_loss_coef = aux_loss_coef
        self.transition_loss_coef = transition_loss_coef

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.model.parameters(), config.lr, eps=config.optim_eps)
        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_aux_losses = []
            log_transition_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_aux_loss = 0
                batch_transition_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.model.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.model.recurrent:
                        dist, value, memory = self.model(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.model(sb.obs)

                    # Compute auxiliary prediction if model has aux head
                    aux_loss = torch.tensor(0.0, device=self.device)
                    if self.aux_loss_coef > 0 and self.model.aux_head is not None:
                        embedding = self.model.compute_embedding(sb.obs)
                        aux_pred = self.model.aux_head(embedding).squeeze(1)
                        aux_loss = torch.nn.functional.mse_loss(aux_pred, sb.chained_distance)

                    # Compute transition prediction loss if model has transition head
                    transition_loss = torch.tensor(0.0, device=self.device)
                    if self.transition_loss_coef > 0 and self.model.transition_head is not None:
                        # Get current raw observation features
                        current_features = sb.obs.features
                        # Predict next raw features
                        action_for_transition = sb.action
                        if len(action_for_transition.shape) == 1:
                            action_for_transition = action_for_transition.unsqueeze(1)
                        predicted_next = self.model.transition_head(current_features, action_for_transition)
                        # Target is the actual next env features (stored during collection)
                        target_next = sb.next_env_features
                        # Only compute loss on valid transitions (episode did not end)
                        valid_mask = sb.transition_valid.bool()
                        if valid_mask.any():
                            transition_loss = torch.nn.functional.mse_loss(
                                predicted_next[valid_mask], target_next[valid_mask]
                            )

                    entropy = dist.entropy().mean()

                    # ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                    # if (len(self.act_shape) == 1):  # Not scalar actions (multivariate)
                    #    delta_log_prob = torch.sum(delta_log_prob, dim=1)
                    ratio = torch.exp(delta_log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    # Add auxiliary loss if enabled
                    if self.aux_loss_coef > 0:
                        loss = loss + self.aux_loss_coef * aux_loss
                    # Add transition loss if enabled
                    if self.transition_loss_coef > 0:
                        loss = loss + self.transition_loss_coef * transition_loss
                    if loss.isnan():
                        print("Loss is NaN")

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
                    batch_transition_loss += transition_loss.item() if isinstance(transition_loss, torch.Tensor) else transition_loss
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.model.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_aux_loss /= self.recurrence
                batch_transition_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                # p.grad can be None if the GNN is not used (e.g. because all assignments only involve a single proposition)
                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.requires_grad and p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad and p.grad is not None],
                                               self.max_grad_norm)
                self.optimizer.step()

                if any(torch.isnan(p).any() for p in self.model.parameters()):
                    print("Model parameters are NaN")

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_aux_losses.append(batch_aux_loss)
                log_transition_losses.append(batch_transition_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "aux_loss": numpy.mean(log_aux_losses),
            "transition_loss": numpy.mean(log_transition_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_steps, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_steps_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i + num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
