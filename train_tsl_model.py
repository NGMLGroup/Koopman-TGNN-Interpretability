import wandb
import os
import torch
import tsl

from dataset.utils import load_FB
from models.DynGraphConvRNN import DynGraphModel
from torch.utils.data import Dataset
from tqdm import tqdm

# Set up config
config = {
        'hidden_size': 16,
        'rnn_layers': 5,
        'readout_layers': 2,
        'cell_type': 'lstm',
        'dim_red': 16,
        'self_loop': False
        }

wandb.init(project="koopman", config=config)
config = wandb.config

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Define dataset
edge_indexes, node_labels, graph_labels = load_FB(config.self_loop)

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
    
dataset = DynGraphDataset(edge_indexes[:50], node_labels[:50], graph_labels[:50])

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Define model
input_size = 1
model = DynGraphModel(
    input_size=input_size,
    hidden_size=config.hidden_size,
    rnn_layers=config.rnn_layers,
    readout_layers=config.readout_layers,
    cell_type=config.cell_type
).to(device)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Set the model to training mode
model.train()

# Train the model
num_epochs = 3
best_loss = float('inf')
patience = 10
counter = 0

for epoch in tqdm(range(num_epochs), desc='Training', position=0, leave=True):
    for data in tqdm(train_dataset, position=1, leave=False):
        inputs, labels = data

        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(inputs.input.x.unsqueeze(0), inputs.edge_index, None)

        # Compute the loss
        loss = criterion(outputs.squeeze(), labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    # Step the scheduler
    scheduler.step()

    # Validation
    total_loss = 0
    with torch.no_grad():
        for data in val_dataset:
            inputs, labels = data

            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, _ = model(inputs.input.x.unsqueeze(0), inputs.edge_index, None)

            # Compute the loss
            loss = criterion(outputs.squeeze(), labels)

            # Accumulate the total loss
            total_loss += loss.item()

    # Calculate the average validation loss
    avg_loss = total_loss / len(val_dataset)

    # Print the average validation loss
    print("Average Validation Loss: {:.4f}".format(avg_loss))

    # Check if the current loss is the best so far
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
    else:
        counter += 1

    # Check if early stopping criteria is met
    if counter >= patience:
        print("Early stopping at epoch", epoch)
        break
        

# Set the model to evaluation mode
model.eval()

# Validation
total_loss = 0
with torch.no_grad():
    for data in tqdm(val_dataset, desc='Validation'):
        inputs, labels = data

        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, _ = model(inputs.input.x.unsqueeze(0), inputs.edge_index, None)

        # Compute the loss
        loss = criterion(outputs.squeeze(), labels)

        # Accumulate the total loss
        total_loss += loss.item()

# Calculate the average validation loss
avg_loss = total_loss / len(val_dataset)

# Print the average validation loss
print("Average Validation Loss: {:.4f}".format(avg_loss))

# make pca and plots

# log on wandb the loss plot and the other relevant plots

# log on tqdm the val loss and lr