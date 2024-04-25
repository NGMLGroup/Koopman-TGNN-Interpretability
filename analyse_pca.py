import wandb
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from dataset.utils import process_classification_dataset


config = {
        'dataset': 'facebook_ct1', # 'infectious_ct1', # 
        'reservoir_size': 32,
        'input_scaling': 0.9,
        'reservoir_layers': 5,
        'leaking_rate': 0.3,
        'spectral_radius': 0.5,
        'density': 0.6,
        'reservoir_activation': 'tanh',
        'alpha_decay': False,
        'add_self_loops': False,
        'b_leaking_rate': True,
        'seed': 42
        }

wandb.init(project="koopman", config=config)
config = wandb.config

seed = config.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# load dataset
dataset, states = process_classification_dataset(config, device, ignore_file=True, verbose=True)
inputs, test_inputs, labels, test_labels = train_test_split(dataset.inputs, dataset.targets, test_size=0.2, random_state=seed)
train_X, val_X, train_y, val_y = train_test_split(states.inputs, states.targets, test_size=0.2, random_state=seed)

# Find accuracy
classifier = LogisticRegression(max_iter=5000, random_state=seed)
classifier.fit(inputs, labels)
y_pred = classifier.predict(test_inputs)
accuracy = accuracy_score(test_labels, y_pred)
wandb.log({'accuracy': accuracy})

# PCA
pc = []
max_accs = []

for dim_red in range(1, train_X.shape[-1]):
    pca4 = PCA(n_components=dim_red)
    # use last time-step so that maybe it converged to something
    train_X_red4 = pca4.fit_transform(train_X[:,-1,:])
    val_X_red4 = pca4.transform(val_X[:,-1,:])
    accs = []

    for i in range(0, train_X_red4.shape[-1]):
        x = train_X_red4[:,i]
        x = x[:, np.newaxis]
        classifier4 = LogisticRegression(max_iter=5000, random_state=seed)
        classifier4.fit(x, train_y)
        x = val_X_red4[:,i]
        x = x[:, np.newaxis]
        y_pred = classifier4.predict(x)
        accs.append(accuracy_score(val_y, y_pred))

    accs = np.array(accs)
    pc.append(np.argmax(accs) + 1)
    max_accs.append(np.max(accs))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(pc, bins=train_X.shape[-1], range=(0, train_X.shape[-1]), alpha=0.5, color='blue')
ax1.set_ylabel('Frequency')

ax2 = ax1.twinx()
ax2.plot(max_accs, color='red')
ax2.set_ylabel('Max accuracy')
ax1.set_xlabel('Number of PCs')
plt.tight_layout()

wandb.log({"hist_PC_img": wandb.Image(fig)})
plt.close(fig)

most_represented = max(set(pc), key=pc.count)
wandb.log({'most_imp_PC': most_represented})