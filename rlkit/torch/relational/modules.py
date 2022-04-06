from torch import nn as nn
from torch.nn import Parameter, functional as F

from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp
from rlkit.torch.relational.relational_util import *


class MessagePassing(PyTorchModule):
    """
    Message passing
    """
    def __init__(self,
                 embedding_dim,
                 adjacency_matrix=None,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 softmax_temperature=1.0):
        # self.save_init_params(locals())
        super().__init__()
        self.adjacency = adjacency_matrix #ptu.from_numpy(grid_adj(25))
        self.fc_createheads = nn.Linear(embedding_dim, embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))

        self.activation_fnx = activation_fnx

    def forward(self, vertices, mask, return_attention=False):
        """
        N, nV, nE memory -> N, nV, nE updated memory

        :param vertices:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = vertices.size()
        # assert len(query.size()) == 3

        nH=1

        adj = self.adjacency(nQ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, nQ, nQ, 1, 1)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(adj)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        adj_probs = F.softmax(adj / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)


        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = vertices.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        message_passing_result = (memory * adj_probs * memory_mask).sum(2).squeeze(2)

        message_passing_result = self.activation_fnx(message_passing_result)
        adj_matrix=None
        if return_attention:
            adj_matrix = adj_probs.reshape((nQ, nV))
        return message_passing_result, adj_matrix
        # return attention_result

class Attention(PyTorchModule):
    """
    Additive, multi-headed attention
    """
    def __init__(self,
                 embedding_dim,
                 num_heads=1,
                 layer_norm=True,
                 activation_fnx=F.leaky_relu,
                 softmax_temperature=1.0,
                 residual_connection=False):
        # self.save_init_params(locals())
        super().__init__()
        self.fc_createheads = nn.Linear(embedding_dim, num_heads * embedding_dim)
        self.fc_logit = nn.Linear(embedding_dim, 1)
        self.fc_reduceheads = nn.Linear(num_heads * embedding_dim, embedding_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(i) for i in [num_heads*embedding_dim, 1, embedding_dim]]) if layer_norm else None
        self.softmax_temperature = Parameter(torch.tensor(softmax_temperature))
        self.residual_connection=residual_connection
        self.activation_fnx = activation_fnx

    def forward(self, query, context, memory, mask, return_attention=False):
        """
        N, nV, nE memory -> N, nV, nE updated memory

        :param query:
        :param context:
        :param memory:
        :param mask: N, nV
        :return:
        """
        N, nQ, nE = query.size()
        # assert len(query.size()) == 3

        # assert self.fc_createheads.out_features % nE == 0
        nH = int(self.fc_createheads.out_features / nE)

        nV = memory.size(1)
        if self.residual_connection:
            residual_values= memory.clone()
        # assert len(mask.size()) == 2

        # N, nQ, nE -> N, nQ, nH, nE
        # if nH > 1:
        query = self.fc_createheads(query).view(N, nQ, nH, nE)
        # else:
        #     query = query.view(N, nQ, nH, nE)

        # if self.layer_norms is not None:
        #     query = self.layer_norms[0](query)
        # N, nQ, nH, nE -> N, nQ, nV, nH, nE
        query = query.unsqueeze(2).expand(-1, -1, nV, -1, -1)

        # N, nV, nE -> N, nQ, nV, nH, nE
        context = context.unsqueeze(1).unsqueeze(3).expand_as(query)

        # -> N, nQ, nV, nH, 1
        qc_logits = self.fc_logit(torch.tanh(context + query))

        # if self.layer_norms is not None:
        #     qc_logits = self.layer_norms[1](qc_logits)

        # N, nV -> N, nQ, nV, nH, 1
        logit_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(-1).expand_as(qc_logits)

        # qc_logits N, nQ, nV, nH, 1 -> N, nQ, nV, nH, 1
        attention_probs = F.softmax(qc_logits / self.softmax_temperature * logit_mask + (-99999) * (1 - logit_mask), dim=2)


        # N, nV, nE -> N, nQ, nV, nH, nE
        memory = memory.unsqueeze(1).unsqueeze(3).expand(-1, nQ, -1, nH, -1)

        # N, nV -> N, nQ, nV, nH, nE
        memory_mask = mask.unsqueeze(1).unsqueeze(3).unsqueeze(4).expand_as(memory)

        # assert memory.size() == attention_probs.size() == mask.size(), (memory.size(), attention_probs.size(), memory_mask.size())

        # N, nQ, nV, nH, nE -> N, nQ, nH, nE
        attention_heads = (memory * attention_probs * memory_mask).sum(2).squeeze(2)

        attention_heads = self.activation_fnx(attention_heads)
        # N, nQ, nH, nE -> N, nQ, nE
        # if nQ > 1:
        attention_result = self.fc_reduceheads(attention_heads.view(N, nQ, nH*nE))
        # else:
        #     attention_result = attention_heads.view(N, nQ, nE)

        # attention_result = self.activation_fnx(attention_result)
        #TODO: add nonlinearity here...

        # if self.layer_norms is not None:
        #     attention_result = self.layer_norms[2](attention_result)

        # assert len(attention_result.size()) == 3
        if self.residual_connection:
            attention_result = attention_result + residual_values
        attention_matrix=None
        if return_attention:
            attention_matrix = attention_probs.squeeze()
        return attention_result

class GraphToGraph(PyTorchModule):
    """
    Uses attention to perform message passing between 1-hop neighbors given the adjacency matrix
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True, adjacency_matrix=None,
                 **kwargs):
        # self.save_init_params(locals())
        super().__init__()
        # self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.messagepassing = MessagePassing(embedding_dim, adjacency_matrix, layer_norm=layer_norm)
        self.layer_norm= nn.LayerNorm(embedding_dim) if layer_norm else None

    def forward(self, vertices, mask, return_attention=False):
        """

        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        assert len(vertices.size()) == 3
        N, nV, nE = vertices.size()

        if self.layer_norm is not None:
            new_vertices, attention_matrix = self.messagepassing(vertices, mask, return_attention=return_attention)
            new_vertices = self.layer_norm(new_vertices)
        else:
            new_vertices, attention_matrix = self.messagepassing(vertices, mask, return_attention=return_attention)

        return new_vertices, attention_matrix



class AttentiveGraphToGraph(PyTorchModule):
    """
    Uses attention to perform message passing between 1-hop neighbors in a fully-connected graph
    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 layer_norm=True,
                 **kwargs):
        # self.save_init_params(locals())
        super().__init__()
        self.fc_qcm = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)
        self.layer_norm= nn.LayerNorm(3*embedding_dim) if layer_norm else None

    def forward(self, vertices, mask):
        """

        :param vertices: N x nV x nE
        :return: updated vertices: N x nV x nE
        """
        assert len(vertices.size()) == 3
        N, nV, nE = vertices.size()
        assert mask.size() == torch.Size([N, nV]), f"mask {mask.size()} not equal to {[N, nV]}"

        # -> (N, nQ, nE), (N, nV, nE), (N, nV, nE)

        # if self.layer_norm is not None:
        #     qcm_block = self.layer_norm(self.fc_qcm(vertices))
        # else:
        qcm_block = self.fc_qcm(vertices)

        query, context, memory = qcm_block.chunk(3, dim=-1)

        return self.attention(query, context, memory, mask)


class AttentiveGraphPooling(PyTorchModule):
    """
    Pools nV vertices to a single vertex embedding

    """
    def __init__(self,
                 embedding_dim=64,
                 num_heads=1,
                 init_w=3e-3,
                 layer_norm=True,
                 mlp_kwargs=None):
        # self.save_init_params(locals())
        super().__init__()
        self.fc_cm = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        self.input_independent_query = Parameter(torch.Tensor(embedding_dim))
        self.input_independent_query.data.uniform_(-init_w, init_w)
        # self.num_heads = num_heads
        self.attention = Attention(embedding_dim, num_heads=num_heads, layer_norm=layer_norm)

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None

    def forward(self, vertices, mask):
        """
        N, nV, nE -> N, nE
        :param vertices:
        :param mask:
        :return:
        """
        N, nV, nE = vertices.size()

        # nE -> N, nQ, nE where nQ == self.num_heads
        query = self.input_independent_query.unsqueeze(0).unsqueeze(0).expand(N, 1, -1)

        # if self.layer_norm is not None:
        #     cm_block = self.layer_norm(self.fc_cm(vertices))
        # else:
        # cm_block = self.fc_cm(vertices)
        # context, memory = cm_block.chunk(2, dim=-1)
        context = vertices
        memory = vertices

        # gt.stamp("Readout_preattention")
        attention_result = self.attention(query, context, memory, mask)

        # gt.stamp("Readout_postattention")
        # return attention_result.sum(dim=1) # Squeeze nV dimension so that subsequent projection function does not have a useless 1 dimension
        if self.proj is not None:
            return self.proj(attention_result).squeeze(1)
        else:
            return attention_result

class MaxGraphPooling(PyTorchModule):
    """
    Pools nV vertices to a single vertex embedding with Maximum

    """
    def __init__(self,
                 mlp_kwargs=None):
        # self.save_init_params(locals())
        super().__init__()

        # self.layer_norm = nn.LayerNorm(2*embedding_dim) if layer_norm else None

        if mlp_kwargs is not None:
            self.proj = Mlp(**mlp_kwargs)
        else:
            self.proj = None

    def forward(self, vertices, mask, return_attention=False):
        """
        N, nV, nE -> N, nE
        :param vertices:
        :param mask:
        :return:
        """
        N, nV, nE = vertices.size()

        max_pooled_vertices, max_indexes = torch.max(vertices, 1)

        if self.proj is not None:
            attention_result = self.proj(max_pooled_vertices).squeeze(1)

        return attention_result


class GraphPropagation(PyTorchModule):
    """
    Input: state
    Output: context vector
    """

    def __init__(self,
                 num_relational_blocks=1,
                 num_query_heads=1,
                 graph_module_kwargs=None,
                 layer_norm=False,
                 activation_fnx=F.leaky_relu,
                 graph_module=AttentiveGraphToGraph,
                 post_residual_activation=True,
                 recurrent_graph=False,
                 **kwargs
                 ):
        """

        :param embedding_dim:
        :param lstm_cell_class:
        :param lstm_num_layers:
        :param graph_module_kwargs:
        :param style: OSIL or relational inductive bias.
        """
        # self.save_init_params(locals())
        super().__init__()

        # Instance settings

        self.num_query_heads = num_query_heads
        self.num_relational_blocks = num_relational_blocks
        assert graph_module_kwargs, graph_module_kwargs
        self.embedding_dim = graph_module_kwargs['embedding_dim']

        if recurrent_graph:
            rg = graph_module(**graph_module_kwargs)
            self.graph_module_list = nn.ModuleList(
                [rg for i in range(num_relational_blocks)])
        else:
            self.graph_module_list = nn.ModuleList(
                [graph_module(**graph_module_kwargs) for i in range(num_relational_blocks)])

        # Layer norm takes in N x nB x nE and normalizes
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embedding_dim) for i in range(num_relational_blocks)])

        # What's key here is we never use the num_objects in the init,
        # which means we can change it as we like for later.

        """
        ReNN Arguments
        """
        self.layer_norm = layer_norm
        self.activation_fnx = activation_fnx

    def forward(self, vertices, mask=None, *kwargs):
        """

        :param shared_state: state that should be broadcasted along nB dimension. N * (nR + nB * nF)
        :param object_and_goal_state: individual objects
        :return:
        """
        output = vertices

        for i in range(self.num_relational_blocks):
            new_output = self.graph_module_list[i](output, mask)
            new_output = output + new_output

            output = self.activation_fnx(new_output) # Diff from 7/22
            # Apply layer normalization
            if self.layer_norm:
                output = self.layer_norms[i](output)
        return output

