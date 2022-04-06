from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt
from rlkit.torch.core import np_to_pytorch_batch

from rlkit.torch.optim.mpi_adam import MpiAdam

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss',
)


class RePReLSACTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            operator_policies,
            operator_qf1s,
            operator_qf2s,
            operator_target_qf1s,
            operator_target_qf2s,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            gpu_id=0
    ):
        super().__init__()
        self.gpu_id = gpu_id
        self.num_operators = len(operator_policies)
        self.env = env
        self.operator_policies = operator_policies
        self.operator_qf1s = operator_qf1s
        self.operator_qf2s = operator_qf2s
        self.operator_target_qf1s = operator_target_qf1s
        self.operator_target_qf2s = operator_target_qf2s
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = {operator: ptu.zeros(1, requires_grad=True) for operator, policy in operator_policies.items()}
            self.alpha_optimizers = {operator: optimizer_class(
                [self.log_alpha[operator]],
                lr=policy_lr,
                gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
            ) for operator, policy in operator_policies.items()}

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizers = {operator: optimizer_class(
            policy.parameters(),
            lr=policy_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        ) for operator, policy in self.operator_policies.items()}
        self.qf1_optimizers = {operator: optimizer_class(
            qf1.parameters(),
            lr=qf_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        ) for operator, qf1 in self.operator_qf1s.items()}
        self.qf2_optimizers = {operator: optimizer_class(
            qf2.parameters(),
            lr=qf_lr,
            gpu_id=self.gpu_id if ptu.get_mode() == "gpu_opt" else None
        ) for operator, qf2 in self.operator_qf2s.items()}

        if optimizer_class is MpiAdam:
            for operator, qf1 in self.qf1_optimizers.items(): qf1.sync()
            for operator, qf2 in self.qf2_optimizers.items(): qf2.sync()
            for operator, alpha in self.alpha_optimizers.items(): alpha.sync()
            for operator, policy in self.policy_optimizers.items(): policy.sync()

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train(self, np_batch_data):
        self._num_train_steps += 1
        batch = {}
        for operator, np_batch in np_batch_data.items():
            batch[operator] = np_to_pytorch_batch(np_batch)
        return self.train_from_torch(batch)


    def train_from_torch(self, operator_batch):
        gt.blank_stamp()
        stats={}
        for operator, batch in operator_batch.items():
            qf1 = self.operator_qf1s[operator]
            qf2 = self.operator_qf2s[operator]
            policy = self.operator_policies[operator]
            target_qf1 = self.operator_target_qf1s[operator]
            target_qf2 = self.operator_target_qf2s[operator]

            qf1_optimizer = self.qf1_optimizers[operator]
            qf2_optimizer = self.qf2_optimizers[operator]
            alpha_optimizer = self.alpha_optimizers[operator]
            policy_optimizer = self.policy_optimizers[operator]

            losses, _stats = self.compute_loss(
                batch,
                skip_statistics=not self._need_to_update_eval_statistics,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                target_qf1=target_qf1,
                target_qf2=target_qf2,
                operator=operator
            )
            stats[operator]=_stats
            """
            Update networks
            """
            if self.use_automatic_entropy_tuning:
                alpha_optimizer.zero_grad()
                losses.alpha_loss.backward()
                alpha_optimizer.step()

            policy_optimizer.zero_grad()
            losses.policy_loss.backward()
            policy_optimizer.step()

            qf1_optimizer.zero_grad()
            losses.qf1_loss.backward()
            qf1_optimizer.step()

            qf2_optimizer.zero_grad()
            losses.qf2_loss.backward()
            qf2_optimizer.step()

            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    qf1, target_qf1, self.soft_target_tau
                )
                ptu.soft_update_from_to(
                    qf2, target_qf2, self.soft_target_tau
                )

        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        self._n_train_steps_total += 1
        gt.stamp('sac training', unique=False)

    # def update_target_networks(self):
    #     ptu.soft_update_from_to(
    #         self.qf1, self.target_qf1, self.soft_target_tau
    #     )
    #     ptu.soft_update_from_to(
    #         self.qf2, self.target_qf2, self.soft_target_tau
    #     )

    def compute_loss(
            self,
            batch,
            skip_statistics=False,
            policy=None,
            qf1=None,
            qf2=None,
            target_qf1=None,
            target_qf2=None,
            operator=None
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """
        dist = policy(obs)
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha[operator] * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha[operator].exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            qf1(obs, new_obs_actions),
            qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = qf1(obs, actions)
        q2_pred = qf2(obs, actions)
        next_dist = policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            target_qf1(next_obs, new_next_actions),
            target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics[f'{operator}/QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics[f'{operator}/QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics[f'{operator}/Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                f'{operator}/Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                f'{operator}/Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                f'{operator}/Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                f'{operator}/Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics[f'{operator}/Alpha'] = alpha.item()
                eval_statistics[f'{operator}/Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        for operator, _ in self.operator_qf1s.items():
            stats.update(self.eval_statistics[operator])
        #stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return list(self.operator_policies.values()) + \
               list(self.operator_qf1s.values()) + \
               list(self.operator_qf2s.values()) + \
               list(self.operator_target_qf1s.values()) + \
               list(self.operator_target_qf2s.values())

    @property
    def optimizers(self):
        return list(self.policy_optimizers.values()) \
               + list(self.qf1_optimizers.values()) \
               + list(self.qf2_optimizers.values()) \
               + list(self.alpha_optimizers.values())

    def get_snapshot(self):
        return dict(
            operator_policies=self.operator_policies,
            operator_qf1s=self.operator_qf1s,
            operator_qf2s=self.operator_qf2s,
            operator_target_qf1s=self.operator_target_qf1s,
            operator_target_qf2s=self.operator_target_qf2s,
            log_alpha=self.log_alpha,
            optimizers=dict(
                alpha_optimizers=self.alpha_optimizers,
                qf1_optimizers=self.qf1_optimizers,
                qf2_optimizers=self.qf2_optimizers,
                policy_optimizers=self.policy_optimizers,
            )
        )
