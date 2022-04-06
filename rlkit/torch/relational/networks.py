from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.relational.InputModules import *
from rlkit.torch.relational.modules import *
from rlkit.torch.sac.policies import FlattenTanhGaussianPolicy, TorchStochasticPolicy

from rlkit.torch.core import eval_np, elem_or_tuple_to_numpy, torch_ify
from rlkit.torch import pytorch_util as ptu

class ValueReNN(PyTorchModule):
    def __init__(self,
                 graph_propagation,
                 readout,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 state_preprocessing_fnx=fetch_preprocessing,
                 *args,
                 value_mlp_kwargs=None,
                 mask=None,
                 composite_normalizer=None,
                 **kwargs):
        # self.save_init_params(locals())
        super().__init__()
        self.input_module = input_module(**input_module_kwargs)
        self._mask = mask
        self.graph_propagation = graph_propagation
        self.readout = readout
        self.composite_normalizer = composite_normalizer

    def forward(self,
                obs,
                mask=None,
                return_stacked_softmax=False):
        if mask is None:
            mask = torch.ones((obs.size()[0], self._mask)).to(ptu.device)
        vertices = self.input_module(obs, mask=mask)
        new_vertices = self.graph_propagation.forward(vertices, mask=mask)
        pooled_output = self.readout(new_vertices, mask=mask)
        return pooled_output


class QValueReNN(PyTorchModule):
    """
    Used for q-value network
    """

    def __init__(self,
                 graph_propagation,
                 readout,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 state_preprocessing_fnx=None,
                 *args,
                 mask=None,
                 composite_normalizer=None,
                 **kwargs):
        # self.save_init_params(locals())
        super().__init__()
        self.graph_propagation = graph_propagation
        self.state_preprocessing_fnx = state_preprocessing_fnx
        self.readout = readout
        self._mask = mask
        self.composite_normalizer = composite_normalizer
        self.input_module = input_module(**input_module_kwargs)

    def forward(self, obs, actions, mask=None, return_stacked_softmax=False):
        # if mask is None:
        #     mask = torch.ones((obs.size()[0], self._mask)).to(ptu.device)
        vertices = self.input_module(obs, actions=actions, mask=mask)
        mask = torch.ones((obs.size()[0], vertices.shape[1])).to(ptu.device)
        relational_block_embeddings = self.graph_propagation.forward(vertices, mask=mask)
        pooled_output = self.readout(relational_block_embeddings, mask=mask)
        assert pooled_output.size(-1) == 1
        return pooled_output


class PolicyReNN(PyTorchModule, TorchStochasticPolicy):
    """
    Used for policy network
    """

    def __init__(self,
                 graph_propagation,
                 readout,
                 *args,
                 input_module=FetchInputPreprocessing,
                 input_module_kwargs=None,
                 mlp_class=FlattenTanhGaussianPolicy,
                 composite_normalizer=None,
                 batch_size=None,
                 mask=None,
                 **kwargs):
        # self.save_init_params(locals())
        super().__init__()
        self.composite_normalizer = composite_normalizer

        # Internal modules
        self.graph_propagation = graph_propagation
        self.selection_attention = readout
        self._mask = mask
        self.mlp = mlp_class(**kwargs['mlp_kwargs'])
        self.input_module = input_module(**input_module_kwargs)

    def forward(self,
                obs,
                mask=None,
                demo_normalizer=False,
                **mlp_kwargs):
        # if mask is None:
        #     mask = torch.ones((obs.size()[0], self._mask)).to(ptu.device)
        vertices = self.input_module(obs, mask=mask)
        mask = torch.ones((obs.size()[0], vertices.shape[1])).to(ptu.device)
        response_embeddings = self.graph_propagation.forward(vertices, mask=mask)

        selected_objects = self.selection_attention(
            vertices=response_embeddings,
            mask=mask
        )
        selected_objects = selected_objects.squeeze(1)
        return self.mlp(selected_objects, **mlp_kwargs)

    def get_action(self,
                   obs_np,
                   **kwargs):
        assert len(obs_np.shape) == 1
        actions, agent_info = self.get_actions(obs_np[None], **kwargs)
        assert isinstance(actions, np.ndarray)
        return actions[0, :], agent_info

    def get_actions(self,
                    obs_np,
                    **kwargs):
        dist = self._get_dist_from_np(obs_np, **kwargs)
        actions = dist.sample()
        agent_info = dict()
        return elem_or_tuple_to_numpy(actions), agent_info

        # mlp_outputs = eval_np(self, obs_np, **kwargs)
        # assert len(mlp_outputs) == 8
        # actions = mlp_outputs[0]
        #
        # agent_info = dict()
        # return actions, agent_info
