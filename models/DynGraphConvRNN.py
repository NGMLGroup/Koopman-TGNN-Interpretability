import torch

from torch import Tensor, nn
from typing import List, Optional, Tuple
from tsl.nn.models import BaseModel
from tsl.nn.layers.recurrent import GraphConvGRUCell, GraphConvLSTMCell
from tsl.nn.layers.recurrent.base import StateType
from tsl.nn.blocks.encoders.recurrent.base import RNNBase


class DynGraphConvRNN(RNNBase):
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
                #  activation: str = None,
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
                #  activation=activation,
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
    
    def forward(self,
                x: Tensor,
                *args,
                h: Optional[List[StateType]] = None,
                **kwargs) -> Tuple[Tensor, List[StateType]]:
        """"""
        # x: [batch, time, *, features]
        if h is None:
            h = self.initialize_state(x)
        elif not isinstance(h, list):
            h = [h]
        # temporal conv
        out = []
        steps = x.size(1)
        edge_indexes, edge_weight = args
        for step in range(steps):
            edge_index = edge_indexes[step] # to allow for time-varying adjacency matrix
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

        if self.return_only_last_state:
            # out: [batch, *, features]
            return out[-1]
        # out: [batch, time, *, features]
        out = torch.stack(out, dim=1)
        out_rec = torch.stack(out_rec, dim=1)
        return out, h, out_rec


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
                 rnn_layers,
                 readout_layers,
                 cell_type='gru',
                 cat_states_layers=False
                #  activation='relu'
                 ):
        super(DynGraphModel, self).__init__()

        self.encoder = DynGraphConvRNN(input_size=input_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cat_states_layers=cat_states_layers,
                           return_only_last_state=False,
                           asymmetric_norm=False,
                           cell=cell_type,
                        #    activation=activation
                           )

        hidden_size = hidden_size * rnn_layers if cat_states_layers else hidden_size
        self.readout = nn.ModuleList()
        for n in range(readout_layers):
            if n < readout_layers - 1:
                self.readout.append(nn.Linear(hidden_size, hidden_size))
            else:
                self.readout.append(nn.Linear(hidden_size, 1))

    def forward(self, x, edge_index, edge_weight, u=None, **kwargs):
        """"""
        # x: [batches steps nodes features]

        x, _, x_rec = self.encoder(x, edge_index, edge_weight)
        h = x
        h_rec = x_rec

        # take last time step and sum over nodes
        x = x[:,-1].sum(-2)
        x_rec = x_rec[:,-1].sum(-2)
        for layer in self.readout:
            x = layer(x)
            x_rec = layer(x_rec)

        return x, h, x_rec, h_rec