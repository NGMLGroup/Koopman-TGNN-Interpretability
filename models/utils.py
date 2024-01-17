"""
Source: https://github.com/Graph-Machine-Learning-Group/sgp/blob/main/tsl/nn/blocks/encoders/gcrnn.py#L43
"""

import torch
import torch_sparse
import numpy as np

from typing import Optional, Union, Tuple, List
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from torch_sparse import SparseTensor
from types import ModuleType


_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
}

def _identity(x):
    return x


def self_normalizing_activation(x: Tensor, r: float = 1.0):
    return r * F.normalize(x, p=2, dim=-1)

def get_functional_activation(activation: Optional[str] = None):
    if activation is None:
        return _identity
    activation = activation.lower()
    if activation == 'linear':
        return _identity
    if activation in ['tanh', 'sigmoid']:
        return getattr(torch, activation)
    if activation in _torch_activations_dict:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")


class _GraphRNN(torch.nn.Module):
    r"""
    Base class for GraphRNNs
    """
    _n_states = None
    hidden_size: int
    _cat_states_layers = False

    def _init_states(self, x):
        assert 'hidden_size' in self.__dict__, \
            f"Class {self.__class__.__name__} must have the attribute " \
            f"`hidden_size`."
        return torch.zeros(size=(self.n_layers, x.shape[0], x.shape[-2], self.hidden_size), device=x.device)

    def single_pass(self, x, h, *args, **kwargs):
        # x: [batch, nodes, channels]
        # h: [layers, batch, nodes, channels]
        h_new = []
        out = x
        for i, cell in enumerate(self.rnn_cells):
            out = cell(out, h[i], *args, **kwargs)
            h_new.append(out)
        return torch.stack(h_new)

    def forward(self, x, *args, h=None, **kwargs):
        # x: [batch, steps, nodes, channels]
        steps = x.size(1)
        if h is None:
            *h, = self._init_states(x)
        if not len(h):
            h = h[0]
        # temporal conv
        out = []
        for step in range(steps):
            h = self.single_pass(x[:, step], h, *args, **kwargs)
            if not isinstance(h, torch.Tensor):
                h_out, _ = h
            else:
                h_out = h
            # append hidden state of the last layer
            if self._cat_states_layers:
                h_out = rearrange(h_out, 'l b n f -> b n (l f)')
            else:
                h_out = h_out[-1]

            out.append(h)
        out = torch.stack(out)
        # out: [steps, layers, batch, nodes, channels]
        # out = rearrange(out, 's b n c -> b s n c')
        # h: [l b n c]
        return out, h


def expand_then_cat(tensors: Union[Tuple[Tensor, ...], List[Tensor]],
                    dim=-1) -> Tensor:
    r"""
    Match the dimensions of tensors in the input list and then concatenate.

    Args:
        tensors: Tensors to concatenate.
        dim (int): Dimension along which to concatenate.
    """
    shapes = [t.shape for t in tensors]
    expand_dims = list(np.max(shapes, 0))
    expand_dims[dim] = -1
    tensors = [t.expand(*expand_dims) for t in tensors]
    return torch.cat(tensors, dim=dim)


def maybe_cat_exog(x, u, dim=-1):
    r"""
    Concatenate `x` and `u` if `u` is not `None`.

    We assume `x` to be a 4-dimensional tensor, if `u` has only 3 dimensions we
    assume it to be a global exog variable.

    Args:
        x: Input 4-d tensor.
        u: Optional exogenous variable.
        dim (int): Concatenation dimension.

    Returns:
        Concatenated `x` and `u`.
    """
    if u is not None:
        if u.dim() == 3:
            u = rearrange(u, 'b s f -> b s 1 f')
        x = expand_then_cat([x, u], dim)
    return x


TensArray = Union[Tensor, np.ndarray]
OptTensArray = Optional[TensArray]
ScipySparseMatrix = Union[coo_matrix, csr_matrix, csc_matrix]
SparseTensArray = Union[Tensor, SparseTensor, np.ndarray, ScipySparseMatrix]

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, np.ndarray):
        return int(edge_index.max()) + 1 if edge_index.size > 0 else 0
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def infer_backend(obj, backend: ModuleType = None):
    if backend is not None:
        return backend
    elif isinstance(obj, Tensor):
        return torch
    elif isinstance(obj, np.ndarray):
        return np
    elif isinstance(obj, SparseTensor):
        return torch_sparse
    raise RuntimeError(f"Cannot infer valid backed from {type(obj)}.")

def weighted_degree(index: TensArray, weights: OptTensArray = None,
                    num_nodes: Optional[int] = None) -> TensArray:
    r"""Computes the weighted degree of a given one-dimensional index tensor.

    Args:
        index (LongTensor): Index tensor.
        weights (Tensor): Edge weights tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    N = maybe_num_nodes(index, num_nodes)
    if isinstance(index, Tensor):
        if weights is None:
            weights = torch.ones((index.size(0)),
                                 device=index.device, dtype=torch.int)
        out = torch.zeros((N,1), dtype=weights.dtype, device=weights.device)
        out.scatter_add_(0, index.unsqueeze(dim=-1), weights.unsqueeze(dim=-1))
    else:
        if weights is None:
            weights = np.ones(index.shape[0], dtype=np.int)
        out = np.zeros(N, dtype=weights.dtype)
        np.add.at(out, index, weights)
    return out


def normalize(edge_index: SparseTensArray, edge_weights: OptTensArray = None,
              dim: int = 0, num_nodes: Optional[int] = None) \
        -> Tuple[SparseTensArray, OptTensArray]:
    r"""Normalize edge weights across dimension :obj:`dim`.

    .. math::
        e_{i,j} =  \frac{e_{i,j}}{deg_{i}\ \text{if dim=0 else}\ deg_{j}}

    Args:
        edge_index (LongTensor): Edge index tensor.
        edge_weights (Tensor): Edge weights tensor.
        dim (int): Dimension over which to compute normalization.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    backend = infer_backend(edge_index)

    if backend is torch_sparse:
        assert edge_weights is None
        deg = edge_index.sum(dim=dim).to(torch.float)
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float('inf')] = 0
        edge_index = deg_inv.view(-1, 1) * edge_index
        return edge_index, None

    index = edge_index[dim]
    degree = weighted_degree(index, edge_weights, num_nodes=num_nodes)
    return edge_index, edge_weights / degree[index].squeeze()