import torch
import numpy as np

from typing import Optional, Union, Tuple, List
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange


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

            out.append(h_out)
        out = torch.stack(out)
        # out: [steps, batch, nodes, channels]
        out = rearrange(out, 's b n c -> b s n c')
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