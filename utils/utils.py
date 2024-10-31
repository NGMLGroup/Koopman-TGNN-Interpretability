import torch
import numpy as np

from einops import rearrange


def get_K(config, states):
    if config['K_type'] == 'model':
        K = get_K_from_model(config)
        emb_engine = None
    elif config['K_type'] == 'data':
        emb_engine, K = get_K_from_data(states, config['dim_red'], config['emb_method'])
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


def get_K_from_data(states, dim_red, method='PCA'):
    from koopman.dmd import DMD

    # compute local Koopman operator
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


def get_K_from_SINDy(edge_index, node_state, dim_red, add_self_dependency=False, degree=2, method='PCA'):
    from koopman.sindy import SINDy

    # Compute Koopman operator with SINDy
    sindy = SINDy(node_state, edge_index, k=dim_red,
                  add_self_dependency=add_self_dependency,
                  degree=degree, emb=method)

    K = sindy.fit()

    # Remove "F" dimension
    sum_K = K.reshape(-1, dim_red, K.shape[1]).sum(axis=1)
    sum_K = sum_K.reshape(K.shape[0], -1, dim_red).sum(axis=2)

    return sum_K


def get_weights_from_SINDy(edge_index, node_state, dim_red, add_self_dependency=False, degree=2, method='PCA'):
    from koopman.sindy import SINDy

    # Compute Koopman operator with SINDy
    # and return edge weights
    sindy = SINDy(node_state, edge_index, k=dim_red, 
                  add_self_dependency=add_self_dependency, 
                  degree=degree, emb=method)
    
    return sindy.compute_weights()


def get_weights_from_DMD(node_state, dim_red, mode_idx=0, method='PCA'):
    from koopman.dmd import DMD

    # Compute Koopman operator with DMD
    node_state = rearrange(node_state, 't n f -> n t f')
    dmd = DMD(node_state, k=dim_red, emb=method)

    return dmd.compute_weights(mode_idx)


def get_weights_from_PCA(node_state, dim_red, method='PCA'):
    from koopman.dmd import DMD

    # Compute Koopman operator with DMD
    node_state = rearrange(node_state, 't n f -> n t f')
    dmd = DMD(node_state, k=dim_red, emb=method)

    return dmd.Zp


def run_saliency(edge_indexes, node_labels, graph_labels, config, device, verbose=False):

    from models.DynGraphConvRNN import DynGraphModel
    from dataset.utils import DynGraphDataset
    from captum.attr import Saliency
    from tqdm import tqdm
    import os
   
    if config['testing'] == True:
        edge_indexes = edge_indexes[:50]
        node_labels = node_labels[:50]
        graph_labels = graph_labels[:50]

    dataset = DynGraphDataset(edge_indexes, node_labels, graph_labels)

    # Define the model
    input_size = 1

    # Remove the unnecessary return values from the forward method
    class RedDynGraphModel(DynGraphModel):
        def forward(self, x, edge_index):
            # Call the forward method of the parent class
            x, _, _, _ = super().forward(x, edge_index, edge_weight=None)
            
            # return only x
            return x
        
    model = RedDynGraphModel(input_size=input_size,
                            hidden_size=config['hidden_size'],
                            rnn_layers=config['rnn_layers'],
                            readout_layers=config['readout_layers'],
                            cell_type=config['cell_type'],
                            cat_states_layers=config['cat_states_layers']).to(device)
    
    # Load the model from the file
    model_filepath = f'models/saved/dynConvRNN_{config["dataset"]}.pt'
    if not os.path.exists(model_filepath):
        raise FileNotFoundError(f"Model file '{model_filepath}' not found. Train the model first.")
    model.load_state_dict(torch.load(model_filepath))
    model = model.to(device)
    model.eval()

    # Define saliency object
    saliency = Saliency(model)

    attributes = []
    labels = []

    if verbose:
        print("Running saliency on model...")

    for data in tqdm(dataset):
        input, label = data

        # Move the inputs and labels to the device
        input = input.to(device)
        label = label.to(device)

        # Compute saliency with respect to input.input.x
        input.input.x.requires_grad = True  
        attribute = saliency.attribute(inputs=input.input.x.unsqueeze(0), target=None, 
                                        additional_forward_args=input.edge_index)
        attributes.append(attribute.squeeze().detach().cpu())
        labels.append(label)
   

    if verbose:
        print("Saliency run complete.")
    
    # List of attributes of shape [T, num_nodes]
    return attributes