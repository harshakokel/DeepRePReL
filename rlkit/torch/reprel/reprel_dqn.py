from rlkit.torch.torch_rl_algorithm import TorchTrainer
import torch.optim as optim
from torch import nn as nn
from collections import OrderedDict
from rlkit.torch.core import np_to_pytorch_batch
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
import numpy as np


class RePReLDQNTrainer(TorchTrainer):
    """
    RePReL with DQN for each option
    """

    def __init__(
            self,
            operator_qfs,
            operator_target_qfs,
            learning_rate=1e-3,
            soft_target_tau=1e-3,
            target_update_period=1,
            qf_criterion=None,
            discount=0.99,
            reward_scale=1.0,
    ):
        super().__init__()
        self.num_operators = len(operator_qfs)
        self.operator_qfs = operator_qfs
        self.operator_target_qfs = operator_target_qfs
        self.learning_rate = learning_rate
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.qf_optimizers = {operator:optim.Adam(
            qf.parameters(),
            lr=self.learning_rate,
        ) for operator, qf in self.operator_qfs.items()}
        self.discount = discount
        self.reward_scale = reward_scale
        self.qf_criterion = qf_criterion or nn.MSELoss()
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return list(self.operator_qfs.values()) \
               + list(self.operator_target_qfs.values())


    def get_snapshot(self):
        return dict(
            operator_qfs=self.operator_qfs,
            operator_target_qfs=self.operator_target_qfs,
        )

    def train(self, np_batch_data):
        self._num_train_steps += 1
        batch = {}
        for operator, np_batch in np_batch_data.items():
            batch[operator] = np_to_pytorch_batch(np_batch)
        return self.train_from_torch(batch)


    def train_from_torch(self, operator_batch):
        for operator, batch in operator_batch.items():
            qf = self.operator_qfs[operator]
            qf_optimizer = self.qf_optimizers[operator]
            target_qf = self.operator_target_qfs[operator]

            rewards = batch['rewards'] * self.reward_scale
            terminals = batch['terminals']
            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']

            """
            Compute loss
            """

            target_q_values = target_qf(next_obs).detach().max(
                1, keepdim=True
            )[0]
            y_target = rewards + (1. - terminals) * self.discount * target_q_values
            y_target = y_target.detach()
            # actions is a one-hot vector
            y_pred = torch.sum(qf(obs) * actions, dim=1, keepdim=True)
            qf_loss = self.qf_criterion(y_pred, y_target)

            """
            Soft target network updates
            """
            qf_optimizer.zero_grad()
            qf_loss.backward()
            qf_optimizer.step()

            """
            Soft Updates
            """
            if self._n_train_steps_total % self.target_update_period == 0:
                ptu.soft_update_from_to(
                    qf, target_qf, self.soft_target_tau
                )

            """
            Save some statistics for eval using just one batch.
            """
            if self._need_to_update_eval_statistics:
                self.eval_statistics[f'{operator}/QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
                self.eval_statistics.update(create_stats_ordered_dict(
                    f'{operator}/Y Predictions',
                    ptu.get_numpy(y_pred),
                ))
        self._n_train_steps_total += 1
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
