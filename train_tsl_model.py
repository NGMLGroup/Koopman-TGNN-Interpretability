import wandb
import os
import torch
import tsl
import random

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from dataset.utils import load_classification_dataset
from models.DynGraphConvRNN import DynGraphModel
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.decomposition import PCA
from einops import rearrange

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Set up config
config = {
        'dataset': 'facebook_ct1', # 'infectious_ct1', #
        'hidden_size': 16,
        'rnn_layers': 5,
        'readout_layers': 2,
        'cell_type': 'lstm',
        'dim_red': 16,
        'self_loop': False,
        'verbose': True,
        'cat_states_layers': True,
        'weight_decay': 1e-4
        }

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
    
dataset = DynGraphDataset(edge_indexes, node_labels, graph_labels)

# Split dataset into train and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, stratify=dataset.graph_labels, random_state=seed)

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
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=config.weight_decay)
# Define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Set the model to training mode
model.train()

# Train the model
num_epochs = 200
best_loss = float('inf')
patience = 30
counter = 0
min_delta = 1e-5

for epoch in tqdm(range(num_epochs), desc='Training', position=0, leave=True):
    for data in tqdm(train_dataset, position=1, leave=False):
        input, label = data

        # Move the inputs and labels to the device
        input = input.to(device)
        label = label.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(input.input.x.unsqueeze(0), input.edge_index, None)

        # Compute the loss
        loss = criterion(output.squeeze(), label)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Step the scheduler
    scheduler.step()
    wandb.log({"lr": scheduler.get_last_lr()[0]})

    # Validation
    total_loss = 0
    hs, labels = [], []
    with torch.no_grad():
        for data in val_dataset:
            input, label = data

            # Move the inputs and labels to the device
            input = input.to(device)
            label = label.to(device)

            # Forward pass
            output, h = model(input.input.x.unsqueeze(0), input.edge_index, None)
            hs.append(h.sum(dim=-2).squeeze()) # sum nodes
            labels.append(label)

            # Compute the loss
            loss = criterion(output.squeeze(), label)

            # Accumulate the total loss
            total_loss += loss.item()

    # Calculate the average validation loss
    avg_loss = total_loss / len(val_dataset)

    # Log the average validation loss
    wandb.log({"epoch": epoch, "val_loss": avg_loss})
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
        

# Set the model to evaluation mode
model.eval()

# Validation
outputs, hs_val, labels_val = [], [], []
with torch.no_grad():
    for data in tqdm(val_dataset, desc='Validation'):
        input, label = data

        # Move the inputs and labels to the device
        input = input.to(device)
        label = label.to(device)

        # Forward pass
        output, h = model(input.input.x.unsqueeze(0), input.edge_index, None)
        outputs.append(output.squeeze())
        hs_val.append(h.sum(dim=-2).squeeze()) # sum nodes
        labels_val.append(label)

# Compute binary classification accuracy
outputs = torch.stack(outputs)
labels_val = torch.stack(labels_val)
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