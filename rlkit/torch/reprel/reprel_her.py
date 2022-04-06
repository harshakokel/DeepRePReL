import torch

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class RePReLHERTrainer(HERTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__(base_trainer)

    def train_from_torch(self, operator_data):
        for operator, data in operator_data.items():
            obs = data['observations']
            next_obs = data['next_observations']
            goals = data['resampled_goals']
            data['observations'] = torch.cat((obs, goals), dim=1)
            data['next_observations'] = torch.cat((next_obs, goals), dim=1)
        self._base_trainer.train_from_torch(operator_data)

    def train(self, np_batch_data):
        self._num_train_steps += 1
        batch = {}
        for operator, np_batch in np_batch_data.items():
            batch[operator] = np_to_pytorch_batch(np_batch)
        return self.train_from_torch(batch)
