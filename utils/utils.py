import torch
import numpy as np


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
    from einops import rearrange

    assert states.ndim == 3, 'States should have shape [batch or nodes, time, features]'
    b, t, f = states.shape
    
    states = rearrange(states, 'b t f -> (b t) f')
    states = emb_engine.transform(states)
    states = rearrange(states, '(b t) f -> b t f', b=b, t=t)
    states = np.dot(states, v)

    return states