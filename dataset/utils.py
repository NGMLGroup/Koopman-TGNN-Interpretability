import torch
import tsl
import h5py
import os

from tsl.datasets import PvUS
from models.DynGraphESN import DynGESNModel
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
from numpy import loadtxt, ndarray
from torch_geometric.utils import add_self_loops



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
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader


def load_FB():
    # Specify the folder path
    folder_path = "dataset/facebook_ct1/"

    # Get the list of file names in the folder
    file_names = ["facebook_ct1_A.txt", "facebook_ct1_edge_attributes.txt", "facebook_ct1_graph_indicator.txt", 
                "facebook_ct1_graph_labels.txt"]

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
    file_path = os.path.join(folder_path, "facebook_ct1_node_labels.txt")
    file = open(file_path, 'r')
    lines = file.readlines()

    lines = [list(map(int, line.strip().split(','))) for line in lines]

    timesteps = 105

    node_label = torch.zeros((timesteps, num_nodes, 1))

    # Assign node labels based on the content of the file
    for n in range(num_nodes):
        if len(lines[n]) == 2 or (len(lines[n]) == 4 and lines[n][0] == lines[n][2]):
            node_label[:, n, 0] = torch.ones((timesteps)) * lines[n][1]
        elif len(lines[n]) == 4:
            node_label[:lines[n][2]-1, n, 0] = torch.ones((lines[n][2]-1)) * lines[n][1]
            node_label[lines[n][2]:, n, 0] = torch.ones((timesteps - lines[n][2])) * lines[n][3]

    # Split node labels based on graph index
    node_labels = [node_label[:, graph_idx == i,:] for i in range(1, num_graphs + 1)]

    edge_index = (A.T - 1).int()
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=timesteps)

    # Split edge index and edge attributes based on graph index
    num_nodes_per_graph = torch.cumsum(torch.cat([torch.tensor([0]), torch.unique(graph_idx, return_counts=True)[1]]), dim=0)
    edge_index_split = [(edge_index[:, (graph_idx[edge_index[0,:]].int() - 1) == i] - num_nodes_per_graph[i]).long() for i in range(0, num_graphs)]
    edge_attr_split = [edge_attr[(graph_idx[edge_index[0,:]].int() - 1) == i] for i in range(0, num_graphs)]

    edge_indexes = []

    # Split edge indexes based on timesteps
    for edge_index, edge_attr in zip(edge_index_split, edge_attr_split):
        edge_indexes.append([edge_index[:, edge_attr == t] for t in range(1, timesteps+1)])
    
    return edge_indexes, node_labels, graph_labels


def run_dyn_gesn_FB(file_path, config, device, verbose=False):
    # Load dataset
    edge_indexes, node_labels, graph_labels = load_FB()

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
                            alpha_decay=config['alpha_decay']).to(device)

    # Save the model to a file
    model_file_path = "models/saved/DynGESN.pt"
    torch.save(model.state_dict(), model_file_path)
    if verbose:
        print(f"Model saved to {model_file_path}")

    inputs = []
    labels = []
    states = []

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
        labels.append(sample.target.y.detach().cpu())

    if verbose:
        print("DynGESN model run complete.")

    # Save the torch_dataset to the H5 file
    if verbose:
        print("Saving results to H5 file...")

    inputs = torch.stack(inputs, dim=0).squeeze()
    states = torch.stack(states, dim=0).squeeze()
    labels = torch.stack(labels, dim=0)

    if not os.path.exists(file_path + "dataset.h5"):
        os.makedirs(file_path + "dataset.h5")

    with h5py.File(file_path + "dataset.h5", "w") as h5_file:
        h5_file.create_dataset("input", data=inputs)
        h5_file.create_dataset("label", data=labels)

    if not os.path.exists(file_path + "states.h5"):
        os.makedirs(file_path + "states.h5")

    with h5py.File(file_path + "states.h5", "w") as h5_file:
        h5_file.create_dataset("states", data=states)
        h5_file.create_dataset("label", data=labels)
    
    if verbose:
        print("Saved results to H5 file.")


def process_FB(config, device, train_ratio=0.7, test_ratio=0.2, batch_size=32, ignore_file=True, verbose=False):

    # Specify the path to the H5 file
    file_path = "dataset/facebook_ct1/processed/"

    if not os.path.exists(file_path + "dataset.h5") or not os.path.exists(file_path + "states.h5") or ignore_file:
        run_dyn_gesn_FB(file_path, config, device, verbose=verbose)
    
    dataset = H5Dataset(file_path + "dataset.h5", "input", "label")
    states = H5Dataset(file_path + "states.h5", "states", "label")
    
    # Calculate the number of samples for each split
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    test_size = int(test_ratio * num_samples)

    # Define the dataloaders for each subset
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, num_samples-test_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(num_samples-test_size, num_samples)))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, val_dataloader, states