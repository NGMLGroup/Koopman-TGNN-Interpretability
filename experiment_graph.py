import torch
import random
import sys
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse

from sklearn.model_selection import train_test_split
from dataset.utils import (load_classification_dataset,
                            process_classification_dataset,
                            ground_truth)
from utils.utils import get_K, change_basis, get_K_from_SINDy
from utils.metrics import *


# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Load configuration from JSON file
config_file = 'configs/GCRN_config.json'
try:
    with open(config_file, 'r') as f:
        configs = json.load(f)
except FileNotFoundError:
    print(f"Error: The configuration file {config_file} was not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON from the configuration file {config_file}.")
    print(f"Details: {e}")
    sys.exit(1)

# Select the dataset
parser = argparse.ArgumentParser(description='Experiment graph')
parser.add_argument('--dataset', type=str, default='infectious_ct1', help='Name of the dataset')

args = parser.parse_args()
dataset_name = args.dataset

# Retrieve the configuration for the selected dataset
config = configs[dataset_name]

seed = config['seed']
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# Load the dataset
dataset, states, node_states, node_labels = process_classification_dataset(config, "DynCRNN", device)
train_X, val_X, train_y, val_y = train_test_split(states.inputs, states.targets, test_size=0.2, random_state=seed)

# Compare with ground-truth labels
nodes_gt, node_sums_gt, times_gt, edges_gt = ground_truth(config['dataset'], config['testing'])
train_node_sums_gt, val_node_sums_gt, train_times_gt, val_times_gt = train_test_split(
    torch.stack(node_sums_gt).numpy(), 
    torch.stack(times_gt).numpy(), 
    test_size=0.2, 
    random_state=seed)


# Time ground truth analysis

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

# Choose eigenvector
mode_idx = 1

r_thr_prec, r_thr_rec, r_thr_f1, r_thr_base = [], [], [], []
r_win_prec, r_win_rec, r_win_f1, r_win_base = [], [], [], []
r_cross, r_corr = [], []
r_mann = []

for g in tqdm(range(len(val_modes)), desc='Time', leave=False):

    if val_y[g]==0 or (val_times_gt[g] == 0).all():
        continue

    fig, thr_prec, thr_rec, thr_f1, thr_base = \
        threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                threshold=config['threshold'],
                                plot=config['plot'])

    if fig is not None:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        if not os.path.exists(f"plots/{config['dataset']}"):
            os.makedirs(f"plots/{config['dataset']}")
        if not os.path.exists(f"plots/{config['dataset']}/time_gt"):
            os.makedirs(f"plots/{config['dataset']}/time_gt")
        if not os.path.exists(f"plots/{config['dataset']}/edge_gt"):
            os.makedirs(f"plots/{config['dataset']}/edge_gt")

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_thr_{mode_idx}.png")
    
    fig, win_prec, win_rec, win_f1, win_base = \
        windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g],
                            window_size=config['window_size'],
                            threshold=config['threshold'],
                            plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_win_{mode_idx}.png")
    
    fig, cc_lag_err, corr = cross_correlation(val_modes[g,:,mode_idx], val_times_gt[g],
                                                plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_cc_{mode_idx}.png")
    
    fig, mw_p_value = mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g], 
                                        window_size=config['window_size'],
                                        plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_mw_{mode_idx}.png")
    
    plt.close('all')
    
    r_thr_prec.append(thr_prec)
    r_thr_rec.append(thr_rec)
    r_thr_f1.append(thr_f1)
    r_thr_base.append(thr_base)
    r_win_prec.append(win_prec)
    r_win_rec.append(win_rec)
    r_win_f1.append(win_f1)
    r_win_base.append(win_base)
    r_cross.append(cc_lag_err)
    r_corr.append(corr)
    r_mann.append(mw_p_value)

# Mann-Whitney U test on whole dataset
fig, mw_p_value_dt = mann_whitney_test_dataset(val_modes[val_y==1,:,mode_idx], 
                                                np.stack(val_times_gt)[val_y==1], 
                                                window_size=config['window_size'],
                                                plot=config['plot'])

if fig is not None:
    fig.savefig(f"plots/{config['dataset']}/time_gt/dataset_mw_{mode_idx}.png")

# Create a dataframe with the results
results = pd.DataFrame({
    'thr_precision': r_thr_prec,
    'thr_recall': r_thr_rec,
    'thr_f1_score': r_thr_f1,
    'thr_baseline_f1': r_thr_base,
    'window_precision': r_win_prec,
    'window_recall': r_win_rec,
    'window_f1_score': r_win_f1,
    'window_baseline_f1': r_win_base,
    'max_corr_lag_error': r_cross,
    'correlation': r_corr,
    'mw_p_value': r_mann,
    'mw_p_value_dt': mw_p_value_dt
})

# Save the dataframe to an Excel file in a new sheet
writer = pd.ExcelWriter(path='results.xlsx', engine='xlsxwriter')
results.to_excel(writer, sheet_name=f"time_gt_{config['dataset']}", index=False)


# Spatial ground truth analysis
aucs = []
edge_indexes, _, _ = load_classification_dataset(config['dataset'], False)
for g in tqdm(range(len(edges_gt)), desc='Topology', leave=False):

    if states.targets[g]==0 or torch.sum(edges_gt[g]) == 0:
        continue

    K = get_K_from_SINDy(edge_indexes[g], node_states[g], config['dim_red'])

    fig, auc = auc_analysis(K, edge_indexes[g], edges_gt[g], plot=config['plot'])
    aucs.append(auc)

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/edge_gt/{g}_mask.png")
    
    plt.close('all')


results = pd.DataFrame({
    'auc': aucs,
})

# Save the dataframe to an Excel file in a new sheet
results.to_excel(writer, sheet_name=f"edge_gt_{config['dataset']}", index=False)
writer.close()