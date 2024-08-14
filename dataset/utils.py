import torch
import tsl
import h5py
import os

import requests
import numpy as np
from io import BytesIO
import warnings

from tsl.datasets import PvUS
from models.DynGraphESN import DynGESNModel
from models.DynGraphConvRNN import DynGraphModel
from tqdm import tqdm
from einops import rearrange
from numpy import loadtxt, ndarray
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset
import sys



class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, input_key, target_key):
        self.file_path = file_path
        self.input_key = input_key
        self.target_key = target_key

        with h5py.File(file_path, "r") as h5_file:
            self.inputs = h5_file[input_key][:]
            self.targets = h5_file[target_key][:]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.inputs[index])
        if isinstance(self.targets[index], ndarray):
            target_data = torch.from_numpy(self.targets[index])
        else:
            target_data = torch.tensor(self.targets[index])
        sample = {'x': input_data, 'y': target_data}
        return input_data, target_data
    

class DynGraphDataset(Dataset):
    def __init__(self, edge_indexes, node_labels, graph_labels):
        self.edge_indexes = edge_indexes
        self.node_labels = node_labels
        self.graph_labels = graph_labels

    def __len__(self):
        return len(self.edge_indexes)

    def __getitem__(self, idx):
        return (tsl.data.data.Data(input={'x': self.node_labels[idx]},
                                  target={'y': self.graph_labels[idx]},
                                  edge_index=self.edge_indexes[idx]),
                self.graph_labels[idx])
    

def run_dyn_gesn_PV(file_path, threshold, config, device, zones=['west'], freq='H', verbose=False):
    """
    Runs the DynGESN model on the PVUS dataset and saves the results to an H5 file.

    Args:
        file_path (str): The path to the H5 file where the results will be saved.
        threshold (float): The threshold value for connectivity.
        config (dict): The configuration parameters for the DynGESN model.
        device (str): The device to run the model on.
        zones (list, optional): The list of zones to consider. Defaults to ['west'].
        freq (str, optional): The frequency of the data. Defaults to 'H'.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.
    """

    dataset = PvUS(root="dataset/pv_us", zones=zones, freq=freq)
    node_index = [i for i in range(0, 200)]
    dataset = dataset.reduce(node_index=node_index)
    sim = dataset.get_similarity("distance")
    connectivity = dataset.get_connectivity(threshold=threshold,
                                            include_self=False,
                                            normalize_axis=1,
                                            layout="edge_index")

    horizon = 24
    torch_dataset = tsl.data.SpatioTemporalDataset(target=dataset.dataframe(),
                                                    connectivity=connectivity,
                                                    mask=dataset.mask,
                                                    horizon=horizon,
                                                    window=1,
                                                    stride=1)

    sample = torch_dataset[0].to(device)

    _, _, feat_size = sample.input.x.shape

    model = DynGESNModel(input_size=feat_size,
                        reservoir_size=config['reservoir_size'],
                        input_scaling=config['input_scaling'],
                        reservoir_layers=config['reservoir_layers'],
                        leaking_rate=config['leaking_rate'],
                        spectral_radius=config['spectral_radius'],
                        density=config['density'],
                        reservoir_activation=config['reservoir_activation'],
                        alpha_decay=config['alpha_decay']).to(device)
    
    
    # Save the model to a file
    model_file_path = "models/saved/DynGESN.pt"
    torch.save(model.state_dict(), model_file_path)
    if verbose:
        print(f"Model saved to {model_file_path}")

    inputs = []
    labels = []

    if verbose:
        print("Running DynGESN model...")

    for i, sample in enumerate(tqdm(torch_dataset)):
        sample = sample.to(device)
        output = model(sample.input.x, sample.input.edge_index, sample.input.edge_weight)[:,:,-1,:]
        inputs.append(output.detach().cpu())
        labels.append(sample.target.y.detach().cpu())

    if verbose:
        print("DynGESN model run complete.")

    # Save the torch_dataset to the H5 file
    if verbose:
        print("Saving results to H5 file...")

    inputs = torch.stack(inputs, dim=0).squeeze()
    inputs = rearrange(inputs, 'b n f -> b n f')
    labels = torch.stack(labels, dim=0)[:, -1, :, :]

    with h5py.File(file_path, "w") as h5_file:
        h5_file.create_dataset("input", data=inputs)
        h5_file.create_dataset("label", data=labels)
    
    if verbose:
        print("Saved results to H5 file.")


def process_PVUS(config, device, threshold=0.7, train_ratio=0.7, test_ratio=0.2, batch_size=32, ignore_file=True, zones=['west'], freq='H', verbose=False):
    """
    Process the PVUS dataset and return dataloaders for training, testing and validation subsets.

    Args:
        config (dict): The configuration parameters for the DynGESN model.
        device (str): The device to run the model on.
        threshold (float): The threshold value for connectivity.
        train_ratio (float, optional): The ratio of samples to be used for training. Defaults to 0.7.
        test_ratio (float, optional): The ratio of samples to be used for testing. Defaults to 0.2.
        batch_size (int, optional): The batch size for the dataloaders. Defaults to 32.
        ignore_file (bool, optional): Whether to ignore the file if it already exists. Defaults to True.
        zones (list, optional): The zones to be processed. Defaults to ['west'].
        freq (str, optional): The frequency of the data. Defaults to 'H'.
        verbose (bool, optional): Whether to print progress messages. Defaults to False.

    Returns:
        tuple: A tuple containing the train dataloader, test dataloader and validation dataloader.
    """

    # Specify the path to the H5 file
    file_path = "dataset/pv_us/processed/" + zones[0] + "_" + freq + ".h5"

    if not os.path.exists(file_path) or ignore_file:
        run_dyn_gesn_PV(file_path, threshold, config, device, zones, freq, verbose=verbose)
    
    dataset = H5Dataset(file_path, "input", "label")
    
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    test_size = int(test_ratio * num_samples)

    # Define the dataloaders for each subset
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, num_samples-test_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(num_samples-test_size, num_samples)))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader


def load_classification_dataset(name, b_add_self_loops=True):
    # Specify the folder path
    folder_path = f"dataset/{name}/"

    # Get the list of file names in the folder
    file_names = [f"{name}_A.txt", f"{name}_edge_attributes.txt", 
                  f"{name}_graph_indicator.txt", f"{name}_graph_labels.txt"]

    # Initialize an empty list to store the data
    data = []

    # Iterate over the file names
    for file_name in file_names:
        # Construct the file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the content of the file
        content = loadtxt(file_path, delimiter=',')
        
        # Convert the content to a torch tensor
        tensor = torch.from_numpy(content)
        
        # Append the tensor to the data list
        data.append(tensor)

    A, edge_attr, graph_idx, graph_labels = data

    num_nodes = A.max().int()
    num_graphs = graph_idx.max().int()

    # Read node labels from file
    file_path = os.path.join(folder_path, f"{name}_node_labels.txt")
    file = open(file_path, 'r')
    lines = file.readlines()

    lines = [list(map(int, line.strip().split(','))) for line in lines]

    timesteps = 106 # FIXME: is it different for each dataset?

    node_label = torch.zeros((timesteps, num_nodes, 1))

    # Assign node labels based on the content of the file
    for n in range(num_nodes):
        if len(lines[n]) == 2 or (len(lines[n]) == 4 and lines[n][0] == lines[n][2]):
            node_label[:, n, 0] = torch.ones((timesteps)) * lines[n][-1]
        elif len(lines[n]) == 4:
            node_label[:lines[n][2]-1, n, 0] = torch.ones((lines[n][2]-1)) * lines[n][1]
            node_label[lines[n][2]:, n, 0] = torch.ones((timesteps - lines[n][2])) * lines[n][3]

    # Split node labels based on graph index
    node_labels = [node_label[:, graph_idx == i,:] for i in range(1, num_graphs + 1)]

    edge_index = (A.T - 1).int()

    # Split edge index and edge attributes based on graph index
    # Node indices need to be translated to 0-based indices
    num_nodes_per_graph = torch.cumsum(
                            torch.cat([
                                torch.tensor([0]), # first graph doesn't need to be translated
                                torch.unique(graph_idx, return_counts=True)[1]
                                ]), 
                            dim=0)
    edge_index_split = [(edge_index[:, (graph_idx[edge_index[0,:].long()].int() - 1) == i] - num_nodes_per_graph[i]).long() for i in range(0, num_graphs)]
    edge_attr_split = [edge_attr[(graph_idx[edge_index[0,:].long()].int() - 1) == i] for i in range(0, num_graphs)]

    edge_indexes = []

    # Split edge indexes based on timesteps
    for edge_index, edge_attr in zip(edge_index_split, edge_attr_split):
        edge_indexes.append([edge_index[:, edge_attr == t] for t in range(1, timesteps+1)])
    
    # Add self-loops for each graph, for each time-step
    if b_add_self_loops:
        for g in tqdm(range(len(edge_indexes))):
            for t in range(len(edge_indexes[g])):
                edge_indexes[g][t], _ = add_self_loops(edge_indexes[g][t], num_nodes=num_nodes_per_graph[g+1])

    return edge_indexes, node_labels, graph_labels


def run_dyn_gesn_classification(file_path, config, device, verbose=False):
    # Load dataset
    edge_indexes, node_labels, graph_labels = load_classification_dataset(config['dataset'], config['add_self_loops'])
    
    if config['testing'] == True:
        edge_indexes = edge_indexes[:50]
        node_labels = node_labels[:50]
        graph_labels = graph_labels[:50]

    # Define the model
    feat_size = 1
    model = DynGESNModel(input_size=feat_size,
                            reservoir_size=config['reservoir_size'],
                            input_scaling=config['input_scaling'],
                            reservoir_layers=config['reservoir_layers'],
                            leaking_rate=config['leaking_rate'],
                            spectral_radius=config['spectral_radius'],
                            density=config['density'],
                            reservoir_activation=config['reservoir_activation'],
                            alpha_decay=config['alpha_decay'],
                            b_leaking_rate=config['b_leaking_rate']).to(device)

    # Save the model to a file
    model_file_path = "models/saved/DynGESN.pt"
    torch.save(model.state_dict(), model_file_path)
    if verbose:
        print(f"Model saved to {model_file_path}")

    inputs = []
    labels = []
    states = []
    node_states = []

    if verbose:
        print("Running DynGESN model...")

    for n in tqdm(range(len(node_labels))):
        sample = tsl.data.data.Data(input={'x': node_labels[n]},
                    target={'y': graph_labels[n]},
                    edge_index=edge_indexes[n]).to(device)
        output = model(sample.input.x, sample.edge_index, None)
        output = rearrange(output, 't n l f -> t n (l f)')
        inputs.append(output.detach().cpu().sum(dim=1)[-1,:])
        states.append(output.detach().cpu().sum(dim=1))
        node_states.append(output.detach().cpu().numpy())
        labels.append(sample.target.y.detach().cpu())

    if verbose:
        print("DynGESN model run complete.")

    # Save the torch_dataset to the H5 file
    if verbose:
        print("Saving results to H5 file...")

    inputs = torch.stack(inputs, dim=0).squeeze()
    states = torch.stack(states, dim=0).squeeze()
    labels = torch.stack(labels, dim=0)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with h5py.File(file_path + "dataset_DynGESN.h5", "w") as h5_file:
        h5_file.create_dataset("input", data=inputs)
        h5_file.create_dataset("label", data=labels)

    with h5py.File(file_path + "states_DynGESN.h5", "w") as h5_file:
        h5_file.create_dataset("states", data=states)
        h5_file.create_dataset("label", data=labels)
    
    if verbose:
        print("Saved results to H5 file.")
    
    return node_states, node_labels


def run_dyn_crnn_classification(file_path, config, device, verbose=False):
    # Load dataset
    edge_indexes, node_labels, graph_labels = load_classification_dataset(config['dataset'], config['add_self_loops'])
    
    if config['testing'] == True:
        edge_indexes = edge_indexes[:50]
        node_labels = node_labels[:50]
        graph_labels = graph_labels[:50]

    dataset = DynGraphDataset(edge_indexes, node_labels, graph_labels)

    # Define the model
    input_size = 1
    model = DynGraphModel(input_size=input_size,
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

    labels = []
    outputs = []
    states = []
    node_states = []

    if verbose:
        print("Running DynCRNN model...")

    for data in tqdm(dataset):
        input, label = data

        # Move the inputs and labels to the device
        input = input.to(device)
        label = label.to(device)

        # Run the model
        x, h, _, _ = model(input.input.x.unsqueeze(0), input.edge_index, None)
        outputs.append(x.squeeze().detach().cpu())
        states.append(h.sum(dim=-2).squeeze().detach().cpu())
        labels.append(label.detach().cpu())
        node_states.append(h.squeeze().detach().cpu())        

    if verbose:
        print("DynCRNN model run complete.")

    # Save the torch_dataset to the H5 file
    if verbose:
        print("Saving results to H5 file...")

    inputs = torch.stack(outputs, dim=0).squeeze() # FIXME: This is the prediction, should I change name?
    states = torch.stack(states, dim=0).squeeze()
    # node_states = np.asarray(node_states)
    labels = torch.stack(labels, dim=0)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with h5py.File(file_path + "dataset_DynCRNN.h5", "w") as h5_file:
        h5_file.create_dataset("input", data=inputs)
        h5_file.create_dataset("label", data=labels)

    with h5py.File(file_path + "states_DynCRNN.h5", "w") as h5_file:
        h5_file.create_dataset("states", data=states)
        h5_file.create_dataset("label", data=labels)

    
    # FIXME: H5Dataset wants homogeneous arrays, should I pad missing nodes with zeros?
    # with h5py.File(file_path + "node_states_DynGESN.h5", "w") as h5_file:
    #     h5_file.create_dataset("node_states", data=node_states)
    #     h5_file.create_dataset("label", data=labels)
    
    if verbose:
        print("Saved results to H5 file.")
    
    return node_states, node_labels


def process_classification_dataset(config, model, device, ignore_file=True, verbose=False):
    # Specify the path to the H5 file
    file_path = f"dataset/{config['dataset']}/processed/"

    cond = not os.path.exists(file_path + f"dataset_{model}.h5") \
            or not os.path.exists(file_path + f"states_{model}.h5") \
            or not os.path.exists(file_path + f"node_states_{model}.h5")
            
    if model=="DynGESN":
        if cond or ignore_file:
            node_states, node_labels = run_dyn_gesn_classification(file_path, config, device, verbose=verbose)
    elif model=="DynCRNN":
        if cond or ignore_file:
            node_states, node_labels = run_dyn_crnn_classification(file_path, config, device, verbose=verbose)
    else:
        raise ValueError(f"Model {model} not supported.")
    
    dataset = H5Dataset(file_path + f"dataset_{model}.h5", "input", "label")
    states = H5Dataset(file_path + f"states_{model}.h5", "states", "label")
    # FIXME: H5Dataset wants homogeneous arrays
    # node_states = H5Dataset(file_path + f"node_states_{model}.h5", "node_states", "label")

    return dataset, states, node_states, node_labels


def ground_truth(dataset_name, testing=False):
    # Load dataset
    edge_indexes, node_labels, _ = load_classification_dataset(dataset_name, False)
    
    if testing == True:
        edge_indexes = edge_indexes[:50]
        node_labels = node_labels[:50]

    nodes_gt, node_sums_gt, times_gt, edges_gt = [], [], [], []

    # Create ground truth explanations
    for node_label, edge_index in tqdm(zip(node_labels, edge_indexes)):
        node_label = node_label.squeeze()

        time_gt = torch.zeros(node_label.shape[0])
        node_gt = torch.zeros_like(node_label)

        for t in range(node_label.shape[0]):
    
            if t != 0:
                node_gt[t,:] = node_gt[t-1,:]

            if t>=node_label.shape[0]-2:
                continue

            # Both nodes of each edge are active at t+2
            cond1 = torch.where(node_label[t+2][edge_index[t].T] == torch.tensor([[1,1]]), 1, 0).all(dim=1)
            # but one of the nodes was not active at t+1
            cond2 = (node_label[t+1][edge_index[t].T][node_label[t+2][edge_index[t].T].all(dim=1)] == torch.tensor([1,0])).all(dim=1)

            if edge_index[t].nelement() != 0 and cond1.any() == True and cond2.any() == True:
                time_gt[t] = cond1.sum().item() / 2
                mask = edge_index[t].T[cond1].flatten()
                node_gt[t,mask] = 1
            else:
                continue

        nodes_gt.append(node_gt)
        node_sums_gt.append(node_gt.squeeze().sum(axis=1).type(torch.int))
        times_gt.append(time_gt)

        # edge ground truth
        edge_index = torch.cat(edge_index, dim=1)
        edge_index = torch.unique(edge_index.T, dim=0).T
        edge_gt = torch.zeros(edge_index.size(1), dtype=torch.int)
        for i in range(edge_index.size(1)):
            src, tgt = edge_index[0, i], edge_index[1, i]
            if src in torch.where(node_gt[-1])[0] and tgt in torch.where(node_gt[-1])[0]:
                edge_gt[i] = 1
        edges_gt.append(edge_gt)
    
    return nodes_gt, node_sums_gt, times_gt, edges_gt



class DataLoader:
    """
    Datasets of multi-variate time-series from paper
    https://arxiv.org/abs/1803.07870
    """

    def __init__(self):
        self.datasets = {
            'AtrialFibrillation': ('https://zenodo.org/records/10852712/files/AF.npz?download=1', 'Multivariate time series classification.\nSamples: 5008 (4823 training, 185 test)\nFeatures: 2\nClasses: 3\nTime series length: 45'),
            'ArabicDigits': ('https://zenodo.org/records/10852747/files/ARAB.npz?download=1', 'Multivariate time series classification.\nSamples: 8800 (6600 training, 2200 test)\nFeatures: 13\nClasses: 10\nTime series length: 93'),
            'Auslan': ('https://zenodo.org/records/10839959/files/Auslan.npz?download=1', 'Multivariate time series classification.\nSamples: 2565 (1140 training, 1425 test)\nFeatures: 22\nClasses: 95\nTime series length: 136'),
            'CharacterTrajectories': ('https://zenodo.org/records/10852786/files/CHAR.npz?download=1', 'Multivariate time series classification.\nSamples: 2858 (300 training, 2558 test)\nFeatures: 3\nClasses: 20\nTime series length: 205'),
            'CMUsubject16': ('https://zenodo.org/records/10852831/files/CMU.npz?download=1', 'Multivariate time series classification.\nSamples: 58 (29 training, 29 test)\nFeatures: 62\nClasses: 2\nTime series length: 580'),
            'ECG2D': ('https://zenodo.org/records/10839881/files/ECG_2D.npz?download=1', 'Multivariate time series classification.\nSamples: 200 (100 training, 100 test)\nFeatures: 2\nClasses: 2\nTime series length: 152'),
            'Japanese_Vowels': ('https://zenodo.org/records/10837602/files/Japanese_Vowels.npz?download=1', 'Multivariate time series classification.\nSamples: 640 (270 training, 370 test)\nFeatures: 12\nClasses: 9\nTime series length: 29'),
            'KickvsPunch': ('https://zenodo.org/records/10852865/files/KickvsPunch.npz?download=1', 'Multivariate time series classification.\nSamples: 26 (16 training, 10 test)\nFeatures: 62\nClasses: 2\nTime series length: 841'),
            'Libras': ('https://zenodo.org/records/10852531/files/LIB.npz?download=1', 'Multivariate time series classification.\nSamples: 360 (180 training, 180 test)\nFeatures: 2\nClasses: 15\nTime series length: 45'),
            'NetFlow': ('https://zenodo.org/records/10840246/files/NET.npz?download=1', 'Multivariate time series classification.\nSamples: 1337 (803 training, 534 test)\nFeatures: 4\nClasses: 2\nTime series length: 997'),
            'RobotArm': ('https://zenodo.org/records/10852893/files/Robot.npz?download=1', 'Multivariate time series classification.\nSamples: 164 (100 training, 64 test)\nFeatures: 6\nClasses: 5\nTime series length: 15'),
            'UWAVE': ('https://zenodo.org/records/10852667/files/UWAVE.npz?download=1', 'Multivariate time series classification.\nSamples: 628 (200 training, 428 test)\nFeatures: 3\nClasses: 8\nTime series length: 315'),
            'Wafer': ('https://zenodo.org/records/10839966/files/Wafer.npz?download=1', 'Multivariate time series classification.\nSamples: 1194 (298 training, 896 test)\nFeatures: 6\nClasses: 2\nTime series length: 198'),
            'Chlorine': ('https://zenodo.org/records/10840284/files/CHLO.npz?download=1', 'Univariate time series classification.\nSamples: 4307 (467 training, 3840 test)\nFeatures: 1\nClasses: 3\nTime series length: 166'), 
            'Phalanx': ('https://zenodo.org/records/10852613/files/PHAL.npz?download=1', 'Univariate time series classification.\nSamples: 539 (400 training, 139 test)\nFeatures: 1\nClasses: 3\nTime series length: 80'),
            'SwedishLeaf': ('https://zenodo.org/records/10840000/files/SwedishLeaf.npz?download=1', 'Univariate time series classification.\nSamples: 1125 (500 training, 625 test)\nFeatures: 1\nClasses: 15\nTime series length: 128'),
        }

    def available_datasets(self, details=False):
        print("Available datasets:\n")
        for alias, (_, description) in self.datasets.items():
            if details:
                print(f"{alias}\n-----------\n{description}\n")
            else:
                print(alias)

    def get_data(self, alias):

        if alias not in self.datasets:
            raise ValueError(f"Dataset {alias} not found.")

        url, _ = self.datasets[alias]
        response = requests.get(url)
        if response.status_code == 200:

            data = np.load(BytesIO(response.content))
            Xtr = data['Xtr']  # shape is [N,T,V]
            if len(Xtr.shape) < 3:
                Xtr = np.atleast_3d(Xtr)
            Ytr = data['Ytr']  # shape is [N,1]
            Xte = data['Xte']
            if len(Xte.shape) < 3:
                Xte = np.atleast_3d(Xte)
            Yte = data['Yte']
            n_classes_tr = len(np.unique(Ytr))
            n_classes_te = len(np.unique(Yte))
            if n_classes_tr != n_classes_te:
                warnings.warn(f"Number of classes in training and test sets do not match for {alias} dataset.")
            print(f"Loaded {alias} dataset.\nNumber of classes: {n_classes_tr}\nData shapes:\n  Xtr: {Xtr.shape}\n  Ytr: {Ytr.shape}\n  Xte: {Xte.shape}\n  Yte: {Yte.shape}")

            return (Xtr, Ytr, Xte, Yte)
        else:
            print(f"Failed to download {alias} dataset.")
            return None