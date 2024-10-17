import torch
import tsl
import h5py
import os

from models.DynGraphConvRNN import DynGraphModel
from tqdm import tqdm
from numpy import loadtxt, ndarray
from torch_geometric.utils import add_self_loops
from torch.utils.data import Dataset



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

    timesteps_dict = {'facebook_ct1': 106, 
                      'infectious_ct1': 50,
                      'dblp_ct1': 48,
                      'highschool_ct1': 205,
                      'mit_ct1': 5566,
                      'tumblr_ct1': 91}
    
    timesteps = timesteps_dict.get(name)
    
    if timesteps is None:
        raise ValueError(f"Dataset {name} not supported.")

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
            
    if model=="DynCRNN":
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