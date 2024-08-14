import torch
import numpy as np

from einops import rearrange


def get_K(config, states):
    if config['K_type'] == 'model':
        K = get_K_from_model(config)
        emb_engine = None
    elif config['K_type'] == 'data':
        emb_engine, K = get_K_from_data(states, config['dim_red'])
    else:
        raise ValueError(f'Unknown Koopman operator type: {config["K_type"]}')
    return emb_engine, K


def get_K_from_model(config):
    from models.DynGraphConvRNN import DynGraphModel

    model = DynGraphModel(input_size=1,
                            hidden_size=config['hidden_size'],
                            rnn_layers=config['rnn_layers'],
                            readout_layers=config['readout_layers'],
                            cell_type=config['cell_type'],
                            cat_states_layers=config['cat_states_layers'])

    # Load the model from the file
    model_filepath = f'models/saved/dynConvRNN_{config["dataset"]}.pt'
    model.load_state_dict(torch.load(model_filepath))

    # Retrieve the trained Koopman operator
    K = model.encoder.K.detach().cpu().numpy()
    return K


def get_K_from_data(states, dim_red):
    from koopman.dmd import DMD

    # compute local Koopman operator
    method = 'PCA'
    dmd = DMD(states, k=dim_red, emb=method)

    K = dmd.compute_KOP()
    emb_engine = dmd.emb_engine
    return emb_engine, K


def change_basis(states, v, emb_engine):

    assert states.ndim == 3, 'States should have shape [batch or nodes, time, features]'
    b, t, f = states.shape
    
    states = rearrange(states, 'b t f -> (b t) f')
    states = emb_engine.transform(states)
    states = rearrange(states, '(b t) f -> b t f', b=b, t=t)
    states = np.dot(states, v)

    return states


def get_K_from_SINDy(edge_index, node_state, dim_red):
    from koopman.sindy import SINDy

    # Compute Koopman operator with SINDy
    method = 'PCA'
    sindy = SINDy(node_state, edge_index, k=dim_red, emb=method)

    K = sindy.fit()

    # Remove "F" dimension
    num_blocks = K.shape[0] // dim_red
    sum_K = np.array([np.sum(K[i*dim_red:(i+1)*dim_red], axis=0) for i in range(num_blocks)])
    num_blocks = K.shape[1] // dim_red
    sum_K = np.stack([np.sum(sum_K[:,i*dim_red:(i+1)*dim_red], axis=1) for i in range(num_blocks)], axis=1)

    return sum_K
