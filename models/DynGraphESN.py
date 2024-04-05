"""

Source: https://github.com/Graph-Machine-Learning-Group/sgp/tree/main
        https://github.com/Graph-Machine-Learning-Group/sgp/blob/main/lib/nn/reservoir/graph_reservoir.py#L96
Code extensively inspired by https://github.com/stefanonardo/pytorch-esn

"""

import numpy as np
import torch
import torch.nn as nn
import torch.sparse
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import torch_sparse
from torch_geometric.utils import add_self_loops, degree

from torch_sparse import SparseTensor

from tsl.ops.connectivity import normalize_connectivity

from einops import rearrange

from models.utils import _GraphRNN, get_functional_activation, self_normalizing_activation, normalize


class GESNLayer(MessagePassing):
    def __init__(self,
                 input_size,
                 hidden_size,
                 spectral_radius=0.9,
                 leaking_rate=0.9,
                 bias=False,
                 density=0.9,
                 in_scaling=1.,
                 bias_scale=1.,
                 activation='tanh',
                 aggr='add',
                 b_leaking_rate=True,
                 requires_grad=False):
        super(GESNLayer, self).__init__(aggr=aggr)
        self.w_ih_scale = in_scaling
        self.b_scale = bias_scale
        self.density = density
        self.hidden_size = hidden_size
        self.alpha = leaking_rate
        self.spectral_radius = spectral_radius
        self.b_leaking_rate = b_leaking_rate

        assert activation in ['tanh', 'relu', 'self_norm', 'identity']
        if activation == 'self_norm':
            self.activation = self_normalizing_activation
        else:
            self.activation = get_functional_activation(activation)

        self.w_ih = nn.Parameter(torch.Tensor(hidden_size, input_size),
                                 requires_grad=requires_grad)
        self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size),
                                 requires_grad=requires_grad)
        if bias is not None:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size),
                                     requires_grad=requires_grad)
        else:
            self.register_parameter('b_ih', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.w_ih.data.uniform_(-1, 1)
        self.w_ih.data.mul_(self.w_ih_scale)

        if self.b_ih is not None:
            self.b_ih.data.uniform_(-1, 1)
            self.b_ih.data.mul_(self.b_scale)

        # init recurrent weights
        self.w_hh.data.uniform_(-1, 1)

        if self.density < 1:
            n_units = self.hidden_size * self.hidden_size
            mask = self.w_hh.data.new_ones(n_units)
            masked_weights = torch.randperm(n_units)[
                             :int(n_units * (1 - self.density))]
            mask[masked_weights] = 0.
            self.w_hh.data.mul_(mask.view(self.hidden_size, self.hidden_size))

        # adjust spectral radius
        abs_eigs = torch.linalg.eigvals(self.w_hh.data).abs()
        self.w_hh.data.mul_(self.spectral_radius / torch.max(abs_eigs))

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j if edge_weight is not None else x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def forward(self, x, h, edge_index, edge_weight=None):
        """This layer expects a normalized adjacency matrix either in
        edge_index or SparseTensor layout."""
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        if edge_index.numel() == 0:
            h_new = h

        else:
            h_new = self.activation(F.linear(x, self.w_ih, self.b_ih) +
                                    self.propagate(edge_index,
                                                x=F.linear(h, self.w_hh),
                                                edge_weight=edge_weight))
            if self.b_leaking_rate:
                h_new = (1 - self.alpha) * h + self.alpha * h_new
            else:
                connected = torch.heaviside(degree(edge_index[0], x.size(1)),
                                            torch.zeros(x.size(1), device=x.device)).to(device=x.device)[None,:,None]
                h_new = connected * h_new + (1 - connected) * h

        return h_new


class DynGraphESN(_GraphRNN):
    _cat_states_layers = True

    def __init__(self,
                 input_size,
                 hidden_size,
                 input_scaling=1.,
                 num_layers=1,
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 density=0.9,
                 activation='tanh',
                 bias=True,
                 alpha_decay=False,
                 b_leaking_rate=True,
                 requires_grad=False):
        super(DynGraphESN, self).__init__()
        self.mode = activation
        self.input_size = input_size
        self.input_scaling = input_scaling
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.alpha_decay = alpha_decay

        layers = []
        alpha = leaking_rate
        for i in range(num_layers):
            layers.append(
                GESNLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    in_scaling=input_scaling,
                    density=density,
                    activation=activation,
                    spectral_radius=spectral_radius,
                    leaking_rate=alpha,
                    b_leaking_rate=b_leaking_rate,
                    requires_grad=requires_grad
                ))
            if self.alpha_decay:
                alpha = np.clip(alpha - 0.1, 0.1, 1.)

        self.rnn_cells = nn.ModuleList(layers)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.rnn_cells:
            layer.reset_parameters()


class DynGESNModel(nn.Module):
    def __init__(self,
                 input_size,
                 reservoir_size,
                 reservoir_layers,
                 leaking_rate,
                 spectral_radius,
                 density,
                 input_scaling,
                 alpha_decay,
                 reservoir_activation='tanh',
                 b_leaking_rate=True,
                 requires_grad=False
                 ):
        super(DynGESNModel, self).__init__()
        self.reservoir = DynGraphESN(input_size=input_size,
                                  hidden_size=reservoir_size,
                                  input_scaling=input_scaling,
                                  num_layers=reservoir_layers,
                                  leaking_rate=leaking_rate,
                                  spectral_radius=spectral_radius,
                                  density=density,
                                  activation=reservoir_activation,
                                  alpha_decay=alpha_decay,
                                  b_leaking_rate=b_leaking_rate,
                                  requires_grad=requires_grad)

    def forward(self, x, edge_index, edge_weight):
        if not isinstance(edge_index, list):
            return self.forward_tensor(x, edge_index, edge_weight)
        else:
            return self.forward_list(x, edge_index, edge_weight)
    

    def forward_tensor(self, x, edge_index, edge_weight):
        # x : [t n f]
        x = rearrange(x, 't n f -> 1 t n f')
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight)
        if not isinstance(edge_index, SparseTensor):
            num_nodes = None #x.shape[0]
            _, edge_weight = normalize(edge_index, edge_weight)
            col, row = edge_index
            edge_index = SparseTensor(row=row, col=col, value=edge_weight,
                                      sparse_sizes=(x.size(-2), x.size(-2)))
        x, h = self.reservoir(x, edge_index, edge_weight)

        h = rearrange(h, 'l b n f -> b n (l f)')
        x = rearrange(x, 's l b n f -> s n b l f')

        # return h # (batch, nodes, layers x reservoir_size)
        return x.squeeze(dim=2) # return all hidden states (steps, nodes, layers, reservoir_size)
    

    def forward_list(self, x, edge_index, edge_weight):
        # To be used when edge_index changes with time
        # - edge_index: list of edge_index for each time step

        # x : [t n f]
        x = rearrange(x, 't n f -> 1 t n f')
        x, h = self.reservoir(x, edge_index, edge_weight)

        h = rearrange(h, 'l b n f -> b n (l f)')
        x = rearrange(x, 's l b n f -> s n b l f')

        return x.squeeze(dim=2) # return all hidden states (steps, nodes, layers, reservoir_size)