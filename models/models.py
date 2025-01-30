import torch

from torch import Tensor, nn
from typing import List, Optional, Tuple
from tsl.nn.models import BaseModel
from tsl.nn.layers.recurrent import (GraphConvGRUCell, GraphConvLSTMCell, 
                                     DCRNNCell, EvolveGCNHCell, EvolveGCNOCell)
from tsl.nn.layers.recurrent.base import StateType
from tsl.nn.blocks.encoders.recurrent.base import RNNBase


class DynRNNBase(RNNBase):
    def forward(self,
                x: Tensor,
                *args,
                h: Optional[List[StateType]] = None,
                **kwargs) -> Tuple[Tensor, List[StateType]]:
        """"""
        # x: [batch, time, *, features]
        if not hasattr(self, 'K'):
            raise AttributeError("Child class must define self.K in its __init__ method")
        if h is None:
            h = self.initialize_state(x)
        elif not isinstance(h, list):
            h = [h]
        # temporal conv
        out = []
        steps = x.size(1)
        edge_indexes, edge_weight = args
        for step in range(steps):
            if isinstance(edge_indexes, list):
                edge_index = edge_indexes[step] # to allow for time-varying adjacency matrix
            else:
                edge_index = edge_indexes
            h_out = h = self.single_pass(x[:, step], h, edge_index, edge_weight, **kwargs)
            # for multi-state rnns (e.g., LSTMs), use first state for readout
            if not isinstance(h_out[0], torch.Tensor):
                h_out = [_h[0] for _h in h_out]
            # append hidden state of the last layer
            if self.cat_states_layers:
                h_out = torch.cat(h_out, dim=-1)
            else:  # or take last layer's state
                h_out = h_out[-1]
            out.append(h_out)

        out_rec = [out[0] @ self.K**step for step in range(steps)]
        out_rec = torch.stack(out_rec, dim=1)

        if self.return_only_last_state:
            # out: [batch, *, features]
            return out[-1]
        # out: [batch, time, *, features]
        out = torch.stack(out, dim=1)

        return out, h, out_rec


class DynGraphConvRNN(DynRNNBase):
    r"""The Graph Convolutional Recurrent Network based on the paper
    `"Structured Sequence Modeling with Graph Convolutional Recurrent Networks"
    <https://arxiv.org/abs/1612.07659>`_ (Seo et al., ICONIP 2017), using
    :class:`~tsl.nn.layers.graph_convs.GraphConv` as graph convolution.

    Modified to accomodate for time-varying adjacency matrix: the input 
    edge_index is a list of edge_index for each time step.

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of units in the hidden state.
        n_layers (int): Number of hidden layers.
            (default: ``1``)
        cat_states_layers (bool): If :obj:`True`, then the states of each layer
            are concatenated along the feature dimension.
            (default: :obj:`False`)
        return_only_last_state (bool): If :obj:`True`, then the ``forward()``
            method returns only the state at the end of the processing, instead
            of the full sequence of states.
            (default: :obj:`False`)
        cell (str): Type of graph recurrent cell that should be use
            (options: ``'gru'``, ``'lstm'``).
            (default: ``'gru'``)
        bias (bool): If :obj:`False`, then the layer will not learn an additive
            bias vector for each gate.
            (default: :obj:`True`)
        asymmetric_norm (bool): If :obj:`True`, then normalize the edge weights
            as :math:`a_{j \rightarrow i} =  \frac{a_{j \rightarrow i}}
            {deg_{i}}`, otherwise apply the GCN normalization.
            (default: :obj:`True`)
        root_weight (bool): If :obj:`True`, then add a filter (with different
            weights) for the root node itself.
            (default :obj:`True`)
        activation (str, optional): Activation function to be used, :obj:`None`
            for identity function (i.e., no activation).
            (default: :obj:`None`)
        cached (bool): If :obj:`True`, then cached the normalized edge weights
            computed in the first call.
            (default :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 cell: str = 'gru',
                 bias: bool = True,
                 asymmetric_norm: bool = True,
                 root_weight: bool = True,
                 cached: bool = False,
                 **kwargs):
        self.input_size = input_size
        self.hidden_size = hidden_size

        if cell == 'gru':
            cell = GraphConvGRUCell
        elif cell == 'lstm':
            cell = GraphConvLSTMCell
        else:
            raise NotImplementedError(f'"{cell}" cell not implemented.')
        
        if asymmetric_norm:
            norm = 'mean'
        else:
            norm = None

        rnn_cells = [
            cell(input_size if i == 0 else hidden_size,
                 hidden_size,
                 norm=norm, # here there is a bug in tsl library
                 root_weight=root_weight,
                 bias=bias,
                 cached=cached,
                 **kwargs) for i in range(n_layers)
        ]
        super(DynGraphConvRNN, self).__init__(rnn_cells, cat_states_layers,
                                           return_only_last_state)
        
        if not self.cat_states_layers:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size, hidden_size)),
                                    requires_grad=True)
        else:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size*n_layers, hidden_size*n_layers)),
                                    requires_grad=True)


class DCRNN(DynRNNBase):
    """The Diffusion Convolutional Recurrent Neural Network from the paper
    `"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting" <https://arxiv.org/abs/1707.01926>`_ (Li et al., ICLR 2018).

    Args:
        input_size: Size of the input.
        hidden_size: Number of units in the hidden state.
        n_layers: Number of layers.
        k: Size of the diffusion kernel.
        root_weight: Whether to learn a separate transformation for the central
            node.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 k: int = 2,
                 root_weight: bool = True,
                 add_backward: bool = True,
                 bias: bool = True,
                 asymmetric_norm: bool = True,):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        rnn_cells = [
            DCRNNCell(input_size if i == 0 else hidden_size,
                      hidden_size,
                      k=k,
                      root_weight=root_weight,
                      add_backward=add_backward,
                      bias=bias) for i in range(n_layers)
        ]
            
        super(DCRNN, self).__init__(rnn_cells, cat_states_layers,
                                    return_only_last_state)
        
        if not self.cat_states_layers:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size, hidden_size)),
                                    requires_grad=True)
        else:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size*n_layers, hidden_size*n_layers)),
                                    requires_grad=True)


class EvolveGCN(nn.Module):
    r"""EvolveGCN encoder from the paper `"EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs"
    <https://arxiv.org/abs/1902.10191>`_ (Pereja et al., AAAI 2020).

    Args:
        input_size (int): Size of the input.
        hidden_size (int): Number of hidden units in each hidden layer.
        n_layers (int): Number of layers in the encoder.
        asymmetric_norm (bool): Whether to consider the input graph as directed.
        variant (str): Variant of EvolveGCN to use (options: 'H' or 'O')
        root_weight (bool): Whether to add a parametrized skip connection.
        cached (bool): Whether to cache normalized edge_weights.
        activation (str): Activation after each GCN layer.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 n_layers,
                 norm,
                 variant='H',
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False,
                 root_weight=False,
                 cached=False,
                 activation='relu'):
        super(EvolveGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cat_states_layers = cat_states_layers
        self.return_only_last_state = return_only_last_state
        self.n_layers = n_layers
        self.rnn_cells = nn.ModuleList()
        if variant == 'H':
            cell = EvolveGCNHCell
        elif variant == 'O':
            cell = EvolveGCNOCell
        else:
            raise NotImplementedError

        for i in range(self.n_layers):
            self.rnn_cells.append(
                cell(in_size=self.input_size if i == 0 else self.hidden_size,
                     out_size=self.hidden_size,
                     norm=norm,
                     activation=activation,
                     root_weight=root_weight,
                     cached=cached))
        
        if not self.cat_states_layers:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size, hidden_size)),
                                    requires_grad=True)
        else:
            self.K = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(hidden_size*n_layers, hidden_size*n_layers)),
                                    requires_grad=True)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # x : b t n f
        out_t = []
        steps = x.size(1)
        h = [None] * len(self.rnn_cells)
        for t in range(steps):
            if isinstance(edge_index, list):
                edge_index_t = edge_index[t] # to allow for time-varying adjacency matrix
            else:
                edge_index_t = edge_index
            out = x[:, t]
            out_c = []
            for c, cell in enumerate(self.rnn_cells):
                out, h[c] = cell(out, h[c], edge_index_t, edge_weight)
                out_c.append(out)
            # append hidden state of the last layer
            if self.cat_states_layers:
                h_out = torch.cat(out_c, dim=-1)
            else:  # or take last layer's state
                h_out = out_c[-1]
            out_t.append(h_out)

        out_rec = [out_t[0] @ self.K**step for step in range(steps)]
        out_rec = torch.stack(out_rec, dim=1)

        if self.return_only_last_state:
            # out_t: [batch, *, features]
            return out_t[-1]
        # out_t: [batch, time, *, features]
        out_t = torch.stack(out_t, dim=1)

        return out_t, out_c, out_rec


class DynGraphModel(BaseModel):
    """
    Simple spatio-temporal model.

    Input time series are encoded in vectors using a DynGraphConvRNN and then 
    decoded using a an MLP.

    Args:
        input_size (int): Input size.
        hidden_size (int): Units in the hidden layers.
        output_size (int): Size of the optional readout.
        rnn_layers (int): Number of recurrent layers in the encoder.
        readout_layers (int): Number of linear layers in the readout.
        cell_type (str, optional): Type of cell that should be used.
            (options: [``'gru'``, ``'lstm'``]).
            (default: ``'gru'``)
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 rnn_layers,
                 readout_layers,
                 k_kernel=None,
                 evolve_variant='H',
                 encoder_type='dyngcrnn',
                 cell_type='gru',
                 cat_states_layers=False
                 ):
        super(DynGraphModel, self).__init__()

        if encoder_type == 'dyngcrnn':
            encoder = DynGraphConvRNN(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cat_states_layers=cat_states_layers,
                           return_only_last_state=False,
                           asymmetric_norm=False,
                           cell=cell_type)
        elif encoder_type == 'dcrnn':
            encoder = DCRNN(input_size=input_size,
                            hidden_size=hidden_size,
                            n_layers=rnn_layers,
                            cat_states_layers=cat_states_layers,
                            return_only_last_state=False,
                            k=k_kernel,
                            root_weight=True,
                            add_backward=True,
                            bias=True)
        elif encoder_type == 'evolvegcn':
            encoder = EvolveGCN(input_size=input_size,
                                hidden_size=hidden_size,
                                n_layers=rnn_layers,
                                norm=None,
                                variant=evolve_variant,
                                cat_states_layers=cat_states_layers,
                                return_only_last_state=False,
                                root_weight=True,
                                cached=False,
                                activation='relu')
        else:
            raise NotImplementedError(f'"{encoder_type}" encoder not implemented.')
            
        self.encoder = encoder

        hidden_size = hidden_size * rnn_layers if cat_states_layers else hidden_size
        self.readout = nn.ModuleList()
        for n in range(readout_layers):
            if n < readout_layers - 1:
                self.readout.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.readout.append(nn.Linear(hidden_size, output_size))

    def forward(self, x, edge_index, edge_weight, batch=None, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]

        x, _, x_rec = self.encoder(x, edge_index, edge_weight)

        # add batch dimension
        if batch is not None:
            batch_size = batch.max().item() + 1
            h = [x[0,:,batch==n,:] for n in range(batch_size)]
            h_rec = [x_rec[0,:,batch==n,:] for n in range(batch_size)]

            x = torch.stack([g[-1].sum(-2) for g in h], dim=0)
            x_rec = torch.stack([g[-1].sum(-2) for g in h_rec], dim=0)

        else:
            h = x
            h_rec = x_rec

            # take last time step and sum over nodes
            x = x[:,-1].sum(-2)
            x_rec = x_rec[:,-1].sum(-2)

        for layer in self.readout:
            x = layer(x)
            x_rec = layer(x_rec)

        return x, h, x_rec, h_rec