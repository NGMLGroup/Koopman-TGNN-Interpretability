import os
import random
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from dataset.utils import (process_classification_dataset,
                            ground_truth)
from utils.utils import get_K, change_basis
from utils.metrics import *

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = {
        'dataset': 'facebook_ct1', # 'infectious_ct1', #
        'hidden_size': 64,
        'rnn_layers': 5,
        'readout_layers': 1,
        'cell_type': 'lstm',
        'dim_red': 64,
        'add_self_loops': False,
        'verbose': True,
        'cat_states_layers': True,
        'K_type': 'data', # 'model',
        'testing': True
        }


# Load the dataset
dataset, states, node_states, node_labels = process_classification_dataset(config, "DynCRNN", device)
train_X, val_X, train_y, val_y = train_test_split(states.inputs, states.targets, test_size=0.2, random_state=seed)

# Compare with ground-truth labels
nodes_gt, node_sums_gt, times_gt = ground_truth(config['dataset'], config['testing'])
train_node_sums_gt, val_node_sums_gt, train_times_gt, val_times_gt = train_test_split(torch.stack(node_sums_gt).numpy(), torch.stack(times_gt).numpy(), test_size=0.2, random_state=seed)

# Compute Koopman operator
emb_engine, K = get_K(config, train_X)

# Compute eigenvalues and eigenvectors
E, V = np.linalg.eig(K)
idx = np.argsort(np.abs(E))[::-1] # sort eigenvalues and eigenvectors
E = E[idx]
V = V[:, idx]
v12 = V[:,0:2].real # first two eigenvectors (real parts)

# Compute the Koopman modes
modes = change_basis(states.inputs, v12, emb_engine)
val_modes = change_basis(val_X, v12, emb_engine)

# Create an empty dataframe
columns = ['thr_precision', 'thr_recall', 'thr_f1_score',
           'window_precision', 'window_recall', 'window_f1_score',
           'max_corr_lag_error',
           'mw_p_value']
df = {}

r_thr_prec, r_thr_rec, r_thr_f1 = [], [], []
r_win_prec, r_win_rec, r_win_f1 = [], [], []
r_cross = []
r_mann = []

mode_idx = 0
for g in tqdm(range(val_modes.shape[0])):
    if val_y[g]==0:
        continue
    thr_dict = threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g])
    thr_prec, thr_rec, thr_f1 = thr_dict[columns[0]], thr_dict[columns[1]], thr_dict[columns[2]]

    win_dict = windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g])
    win_prec, win_rec, win_f1 = win_dict[columns[3]], win_dict[columns[4]], win_dict[columns[5]]

    r_thr_prec.append(thr_prec)
    r_thr_rec.append(thr_rec)
    r_thr_f1.append(thr_f1)
    r_win_prec.append(win_prec)
    r_win_rec.append(win_rec)
    r_win_f1.append(win_f1)

    r_cross.append(cross_correlation(val_modes[g,:,mode_idx], val_times_gt[g])[columns[6]])
    r_mann.append(mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g])[columns[7]])

# Add results to a new row in the dataframe
df['thr_precision'] = np.mean(r_thr_prec)
df['thr_recall'] = np.mean(r_thr_rec)
df['thr_f1_score'] = np.mean(r_thr_f1)
df['window_precision'] = np.mean(r_win_prec)
df['window_recall'] = np.mean(r_win_rec)
df['window_f1_score'] = np.mean(r_win_f1)
df['max_corr_lag_error'] = np.mean(r_cross)
df['mw_p_value'] = np.mean(r_mann)

ml_results = ml_probes(modes, torch.stack(times_gt).numpy(), seed=seed, verbose=True)

for k, v in ml_results.items():
    df[k] = v

print(df)