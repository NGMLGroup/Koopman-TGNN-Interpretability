import wandb
import os
import torch
import tsl
import random
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from dataset.utils import load_classification_dataset
from models.DynGraphConvRNN import DynGraphModel
from torch.utils.data import Dataset
from tsl.data.batch import DisjointBatch
from tqdm import tqdm
from sklearn.decomposition import PCA
from einops import rearrange

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Select the dataset
parser = argparse.ArgumentParser(description='Experiment graph')
parser.add_argument('--dataset', type=str, default='infectious_ct1', help='Name of the dataset')
parser.add_argument('--hidden_size', type=int, default=16, help='Feature dimension')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers')
parser.add_argument('--readout_layers', type=int, default=1, help='Number of readout layers')
parser.add_argument('--dim_red', type=int, default=16, help='Dimensionality reduction')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--step_size', type=int, default=20, help='Early-stop step size')
parser.add_argument('--gamma', type=float, default=0.5, help='Early-stop gamma')
parser.add_argument('--sweep', type=bool, default=False, help='Sweep')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose')
parser.add_argument('--self_loop', type=bool, default=False, help='Self loop')
parser.add_argument('--cell_type', type=str, default='lstm', help='Cell type')
parser.add_argument('--cat_states_layers', type=bool, default=True, help='Concatenation of states')
parser.add_argument('--beta', type=float, default=0.1, help='Weight of the ridge loss')

args = parser.parse_args()
dataset_name = args.dataset

if not args.sweep:
    # Load configuration from JSON file
    config_file = 'configs/GCRN_config.json'
    with open(config_file, 'r') as f:
        configs = json.load(f)
    # Retrieve the configuration for the selected dataset
    if dataset_name not in configs:
        raise ValueError(f"Hyperparameters for dataset {dataset_name} are missing.")
    config = configs[dataset_name]
    
else:
    config = vars(args)

wandb.init(project="koopman", config=config)
config = wandb.config

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

verbose = config.verbose

# Define dataset
edge_indexes, node_labels, graph_labels = load_classification_dataset(config.dataset, config.self_loop)

class DynGraphDataset(Dataset):
    def __init__(self, edge_indexes, node_labels):
        self.edge_indexes = edge_indexes
        self.node_labels = node_labels

    def __len__(self):
        return len(self.edge_indexes)

    def __getitem__(self, idx):
        pattern = dict(x='t n f', y='f')
        return tsl.data.data.Data(input={'x': self.node_labels[idx]},
                                  edge_index=self.edge_indexes[idx],
                                  pattern=pattern)
    
dataset = DynGraphDataset(edge_indexes, node_labels)

# Split dataset into train and validation sets
train_x, val_x, train_y, val_y = train_test_split(dataset, graph_labels, test_size=0.2, stratify=graph_labels, random_state=seed)

batch_size = 16
train_x_batches = [DisjointBatch.from_data_list(train_x[i:i+batch_size]) for i in range(0, len(train_x), batch_size)]
train_y_batches = [train_y[i:i+batch_size] for i in range(0, len(train_y), batch_size)]
val_x_batches = [DisjointBatch.from_data_list(val_x[i:i+batch_size]) for i in range(0, len(val_x), batch_size)]
val_y_batches = [val_y[i:i+batch_size] for i in range(0, len(val_y), batch_size)]

# Define model
input_size = 1
model = DynGraphModel(
    input_size=input_size,
    hidden_size=config.hidden_size,
    rnn_layers=config.rnn_layers,
    readout_layers=config.readout_layers,
    cell_type=config.cell_type,
    cat_states_layers=config.cat_states_layers
).to(device)

# Define loss function and optimizer
criterion_pred = torch.nn.BCEWithLogitsLoss()
criterion_rec = torch.nn.BCEWithLogitsLoss()
criterion_obs = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.weight_decay)
# Define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# Set the model to training mode
model.train()

# Train the model
num_epochs = 200
best_loss = float('inf')
patience = 30
counter = 0
min_delta = 1e-5

for epoch in tqdm(range(num_epochs), desc='Training', position=0, leave=True):
    for x, y in tqdm(zip(train_x_batches, train_y_batches), position=1, leave=False):

        # Move the inputs and labels to the device
        input = x.to(device)
        label = y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        x, h, x_rec, h_rec = model(input.input.x.unsqueeze(0), input.edge_index, edge_weight=None, batch=input.batch)

        # Compute the loss
        l2_reg = config.weight_decay * torch.sum(torch.pow(model.encoder.K, 2))
        loss_pred = criterion_pred(x.squeeze(), label)
        loss_rec = criterion_rec(x_rec.squeeze(), label)
        loss_obs = torch.stack([criterion_obs(h_rec[i], h[i]) for i in range(len(h))]).mean()
        loss_ridge = loss_obs + l2_reg
        loss_sum = loss_pred + loss_rec + config.beta * loss_ridge

        # Backward pass and optimization
        loss_sum.backward()
        optimizer.step()
    
    # Step the scheduler
    scheduler.step()
    wandb.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0]})

    # Validation
    total_loss = 0
    total_loss_pred, total_loss_rec, total_loss_ridge, total_loss_obs = 0, 0, 0, 0
    hs = []
    with torch.no_grad():
        for x, y in zip(val_x_batches, val_y_batches):

            # Move the inputs and labels to the device
            input = x.to(device)
            label = y.to(device)

            # Forward pass
            x, h, x_rec, h_rec = model(input.input.x.unsqueeze(0), input.edge_index, edge_weight=None, batch=input.batch)
            hs += [h_b.sum(dim=-2).squeeze() for h_b in h] # sum nodes

            # Compute the loss
            l2_reg = config.weight_decay * torch.sum(torch.pow(model.encoder.K, 2))
            loss_pred = criterion_pred(x.squeeze(), label)
            loss_rec = criterion_rec(x_rec.squeeze(), label)
            loss_obs = torch.stack([criterion_obs(h_rec[i], h[i]) for i in range(len(h))]).mean()
            loss_ridge = loss_obs + l2_reg
            loss_sum = loss_pred + loss_rec + config.beta * loss_ridge

            # Accumulate the total loss
            total_loss += loss_sum.item()
            total_loss_pred += loss_pred.item()
            total_loss_rec += loss_rec.item()
            total_loss_ridge += loss_ridge.item()
            total_loss_obs += loss_obs.item()

    # Calculate the average validation loss
    avg_loss = total_loss / len(val_x)
    avg_loss_pred = total_loss_pred / len(val_x)
    avg_loss_rec = total_loss_rec / len(val_x)
    avg_loss_ridge = total_loss_ridge / len(val_x)
    avg_loss_obs = total_loss_obs / len(val_x)

    # Log the average validation loss
    wandb.log({"epoch": epoch, "val_loss": avg_loss})
    wandb.log({"epoch": epoch, "val_loss_pred": avg_loss_pred})
    wandb.log({"epoch": epoch, "val_loss_rec": avg_loss_rec})
    wandb.log({"epoch": epoch, "val_loss_ridge": avg_loss_ridge})
    wandb.log({"epoch": epoch, "val_loss_obs": avg_loss_obs})
    if verbose:
        print("Validation Loss: {:.6f}".format(avg_loss))

    # Check if the current loss is the best so far
    if best_loss - avg_loss > min_delta:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1

    # Check if early stopping criteria is met
    if counter >= patience:
        if verbose:
            print("Early stopping at epoch", epoch)
        break

# Save the model
torch.save(model.state_dict(), f'models/saved/dynConvRNN_{config.dataset}.pt')    

# Set the model to evaluation mode
model.eval()

# Validation
outputs, hs_val, labels_val = [], [], []
with torch.no_grad():
    for x, y in tqdm(zip(val_x_batches, val_y_batches), desc='Validation'):

        # Move the inputs and labels to the device
        input = x.to(device)
        label = y.to(device)

        # Forward pass
        x, h, x_rec, h_rec = model(input.input.x.unsqueeze(0), input.edge_index, edge_weight=None, batch=input.batch)
        outputs.append(x.squeeze())
        hs_val += [h_b.sum(dim=-2).squeeze() for h_b in h] # sum nodes
        labels_val.append(label)

# Compute binary classification accuracy
outputs = torch.cat(outputs)
labels_val = torch.cat(labels_val)
predictions = torch.sigmoid(outputs) > 0.5
accuracy = (predictions == labels_val).sum().item() / len(labels_val)
wandb.log({"accuracy": accuracy})
if verbose:
    print("Accuracy: {:.4f}".format(accuracy))


# Perform PCA on the hidden states
# Train states
hs = torch.stack(hs) # shape [batch, time, hidden_size]
hs = hs.cpu().numpy()
# Validation states
hs_val = torch.stack(hs_val) # shape [batch, time, hidden_size]
hs_val = hs_val.cpu().numpy()

# Dimensionality reduction
dim_red = config.dim_red
pca = PCA(n_components=dim_red)
hs_val_red = pca.fit_transform(hs_val.reshape(-1, hs_val.shape[-1]))
hs_val_red = rearrange(hs_val_red, '(b t) f -> b t f', b=hs_val.shape[0], t=hs_val.shape[1], f=dim_red)


# Plots

# Plot covariance matrix of reduced states
fig, ax = plt.subplots()
cov = ax.imshow(hs_val_red.reshape(-1,hs_val_red.shape[-1]).T @ hs_val_red.reshape(-1,hs_val_red.shape[-1]), cmap='viridis')
plt.colorbar(cov, ax=ax)
wandb.log({"cov_img": wandb.Image(fig)})
plt.close(fig)


# Plot state distribution of the first 2 PCA components
idx0 = labels_val.cpu().numpy() == 0
idx1 = labels_val.cpu().numpy() == 1
label_0 = hs_val_red[idx0, -1, :2]
label_1 = hs_val_red[idx1, -1, :2]

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# Plot the first histogram in the first subplot
hist0 = axs[0].hist2d(label_0[:, 0], label_0[:, 1], bins=10, cmap='Blues', alpha=0.6)
axs[0].set_xlabel('PC 0')
axs[0].set_ylabel('PC 1')
axs[0].set_title('2D Histogram - Label 0')
plt.colorbar(hist0[3], ax=axs[0])
# Plot the second histogram in the second subplot
hist1 = axs[1].hist2d(label_1[:, 0], label_1[:, 1], bins=10, cmap='Reds', alpha=0.6)
axs[1].set_xlabel('PC 0')
axs[1].set_ylabel('PC 1')
axs[1].set_title('2D Histogram - Label 1')
plt.colorbar(hist1[3], ax=axs[1])
# Adjust the spacing between subplots
plt.tight_layout()

wandb.log({"hist_PC_img": wandb.Image(fig)})
plt.close(fig)