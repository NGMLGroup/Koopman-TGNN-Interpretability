import torch
import wandb
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from einops import rearrange

from dataset.utils import process_FB
from DMD.dmd import KANN

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

config = {
        'reservoir_size': 16,
        'input_scaling': 1,
        'reservoir_layers': 5,
        'leaking_rate': 0.1,
        'spectral_radius': 1,
        'density': 1,
        'reservoir_activation': 'tanh',
        'alpha_decay': False,
        'add_self_loops': False,
        'b_leaking_rate': True,
        'dim_red': 16
        }

wandb.init(project="koopman", config=config)
config = wandb.config

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Train ESN readout

# Load Facebook dataset and process it with DynGESN
dataset, states = process_FB(config, device, ignore_file=True, verbose=False)

# Train readout on the last time step
inputs, test_inputs, labels, test_labels = train_test_split(dataset.inputs, dataset.targets, test_size=0.2, random_state=seed)

classifier = LogisticRegression(max_iter=5000, random_state=seed) # Create the classifier
classifier.fit(inputs, labels) # Train the classifier
y_pred = classifier.predict(test_inputs) # Test the classifier
accuracy = accuracy_score(test_labels, y_pred) # Compute the accuracy
wandb.log({'readout_acc': accuracy})


# Train a classifier on the Koopman operators

# Load full reservoir states (with time dimension)
X_train, X_test, labels, test_labels = train_test_split(states.inputs, states.targets, test_size=0.3, random_state=seed)

# Dim reduction
dim_red = config.dim_red
pca = PCA(n_components=dim_red)
X_train_red = pca.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
X_test_red = pca.transform(X_test.reshape(-1, X_test.shape[-1]))
X_train_red = rearrange(X_train_red, '(b t) f -> b t f', b = X_train.shape[0], t=X_train.shape[1], f=dim_red)
X_test_red = rearrange(X_test_red, '(b t) f -> b t f', b = X_test.shape[0], t=X_test.shape[1], f=dim_red)

# Compute a per sample Koopman operator
Ks = []
for x in X_train_red:
    kann_ = KANN(x[np.newaxis,:,:], k=0, emb=None)
    KL = kann_.compute_KOP()
    Ks.append(KL.flatten())
Ks = np.stack(Ks, axis=0)

# Train the classifier
Kclassifier = LogisticRegression(max_iter=1000)
Kclassifier.fit(Ks, labels)

# Test the classifier
Ks_te = []
for x in X_test_red:
    kann_ = KANN(x[np.newaxis,:,:], k=0, emb=None)
    KL = kann_.compute_KOP()
    Ks_te.append(KL.flatten())
Ks_te = np.stack(Ks_te, axis=0)
y_pred = Kclassifier.predict(Ks_te)
accuracy = accuracy_score(test_labels, y_pred)
wandb.log({'K_acc': accuracy})


# Plots

# Plot covariance matrix of reduced states
fig, ax = plt.subplots()
cov = ax.imshow(X_train_red.reshape(-1,X_train_red.shape[-1]).T @ X_train_red.reshape(-1,X_train_red.shape[-1]), cmap='viridis')
plt.colorbar(cov, ax=ax)
wandb.log({"cov_img": wandb.Image(fig)})
plt.close(fig)


# Plot state distribution of the first 2 PCA components
idx0 = labels == 0
idx1 = labels == 1
label_0 = X_train_red[idx0, -1, :2]
label_1 = X_train_red[idx1, -1, :2]

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