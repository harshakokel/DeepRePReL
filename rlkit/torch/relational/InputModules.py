import math

import torch.nn as nn
from torch import nn as nn
from torch.nn import Parameter, functional as F
import rlkit.torch.pytorch_util as ptu
import numpy as np
from rlkit.torch.relational.relational_util import fetch_preprocessing_robot

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp
import torch

import gtimer as gt


class ImageInputPreprocessing(PyTorchModule):

    def __init__(self,
                 normalizer,
                 object_total_dim=3,
                 layer_norm=True,
                 CNN_channels=[12, 24],
                 append_grid=True):
        # self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.append_grid = append_grid
        self.conv = nn.Sequential(
            nn.Conv2d(object_total_dim, CNN_channels[0], kernel_size=2, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(CNN_channels[0], CNN_channels[1], kernel_size=2, stride=1, padding=1), nn.ReLU())
        embedding_dim = CNN_channels[1]
        if self.append_grid:
            embedding_dim += 2
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, obs, actions=None, mask=None):
        vertices = self.conv(obs)
        batch, dim, nrow, ncol = vertices.shape
        vertices = vertices.reshape((batch, dim, nrow * ncol)).transpose(1, 2)
        if self.append_grid:
            # self.edge_index = grid(size[0], size[1])
            x_coordinate = ptu.tensor((2 * np.array([x for y in np.arange(nrow) for x in np.arange(ncol)]).reshape(
                (nrow, ncol, 1)) / (nrow)) - 1, dtype=torch.float32)
            y_coordinate = ptu.tensor((2 * np.array([y for y in np.arange(nrow) for x in np.arange(ncol)]).reshape(
                (nrow, ncol, 1)) / (ncol)) - 1, dtype=torch.float32)
            self._xy_grid = torch.cat((y_coordinate, x_coordinate), dim=2).reshape((nrow * ncol, 2))
            vertices = torch.cat([vertices, self._xy_grid.unsqueeze(0).expand(batch, *self._xy_grid.size())], 2)
        if self.layer_norm is not None:
            return self.layer_norm(vertices)
        else:
            return vertices


class VecToGraphInputPreprocessing(PyTorchModule):

    def __init__(self,
                 normalizer,
                 object_dim,
                 shared_dim,
                 embedding_dim,
                 layer_norm=True):
        # self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.shared_dim = shared_dim
        self.object_dim = object_dim
        self.fc_embed = nn.Linear(object_dim + shared_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, obs, actions=None, mask=None):
        batch_size, obs_dim = obs.size()
        num_vertices = int((obs_dim - self.shared_dim) / self.object_dim)
        shared_features = obs.narrow(1, 0, self.shared_dim)
        flat_object_features = obs.narrow(1, self.shared_dim, num_vertices * self.object_dim)
        object_features = flat_object_features.view(batch_size, num_vertices, self.object_dim)
        shared_features = shared_features.unsqueeze(1).expand(batch_size, num_vertices, self.shared_dim)
        vertices = torch.cat((object_features, shared_features), dim=-1).to(ptu.device)
        if self.layer_norm is not None:
            return self.layer_norm(self.fc_embed(vertices))
        else:
            return self.fc_embed(vertices)


class FetchInputPreprocessing(PyTorchModule):
    """
    Used for the Q-value and value function

    Takes in either obs or (obs, actions) in the forward function and returns the same sized embedding for both

    Make sure actions are being passed in!!
    """

    def __init__(self,
                 normalizer,
                 object_total_dim,
                 embedding_dim,
                 layer_norm=True):
        # self.save_init_params(locals())
        super().__init__()
        self.normalizer = normalizer
        self.fc_embed = nn.Linear(object_total_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, obs, actions=None, mask=None):
        vertices = fetch_preprocessing_robot(obs, actions=actions, normalizer=self.normalizer, mask=mask)

        if self.layer_norm is not None:
            return self.layer_norm(self.fc_embed(vertices))
        else:
            return self.fc_embed(vertices)
