import torch
import wandb
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
from utils.utils import (get_K, change_basis, 
                         get_weights_from_SINDy,
                         get_weights_from_DMD)
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

# Select the dataset
parser = argparse.ArgumentParser(description='Experiment graph')
parser.add_argument('--dataset', type=str, default='tumblr_ct1', help='Name of the dataset')
parser.add_argument('--threshold', default='None', help='Threshold for the threshold-based detection')
parser.add_argument('--window_size', type=int, default=5, help='Window size for the windowing analysis')
parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Plot the results')
parser.add_argument('--sweep', action=argparse.BooleanOptionalAction, help='Sweep')

args = parser.parse_args()

# Handle the None case for threshold
if args.threshold.lower() == 'none':
    args.threshold = None
else:
    args.threshold = float(args.threshold)
dataset_name = args.dataset

# Load configuration from JSON file
config_file = 'configs/GCRN_config.json'
with open(config_file, 'r') as f:
    configs = json.load(f)
# Retrieve the configuration for the selected dataset
if dataset_name not in configs:
    raise ValueError(f"Hyperparameters for dataset {dataset_name} are missing.")

if not args.sweep:
    # If it's not a sweep, load from json
    config = configs[dataset_name]
else:
    # If it's a sweep, overwrite json configs with args
    config = configs[dataset_name]
    for key, value in vars(args).items():
        config[key] = value

wandb.init(project="koopman", config=config)

# Check used configs
print("WandB Configuration Summary:")
for key, value in config.items():
    print(f"{key} ({type(value)}): {value}")

seed = config['seed']
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Create the directory to save the plots
if config['plot']:
    if not os.path.exists("plots"):
        os.makedirs("plots")
    if not os.path.exists(f"plots/{config['dataset']}"):
        os.makedirs(f"plots/{config['dataset']}")
    if not os.path.exists(f"plots/{config['dataset']}/time_gt"):
        os.makedirs(f"plots/{config['dataset']}/time_gt")
    if not os.path.exists(f"plots/{config['dataset']}/node_gt/global_dmd"):
        os.makedirs(f"plots/{config['dataset']}/node_gt/global_dmd")
    if not os.path.exists(f"plots/{config['dataset']}/node_gt/local_dmd"):
        os.makedirs(f"plots/{config['dataset']}/node_gt/local_dmd")
    if not os.path.exists(f"plots/{config['dataset']}/edge_gt/deg2"):
        os.makedirs(f"plots/{config['dataset']}/edge_gt/deg2")
    if not os.path.exists(f"plots/{config['dataset']}/edge_gt/deg3"):
        os.makedirs(f"plots/{config['dataset']}/edge_gt/deg3")


# Load the dataset
dataset, states, node_states, node_labels = process_classification_dataset(config, "DynCRNN", device)
train_X, val_X, train_y, val_y, train_nodes, val_nodes = train_test_split(states.inputs, states.targets, node_states,
                                                  test_size=0.2, random_state=seed)
edge_indexes, _, _ = load_classification_dataset(config['dataset'], False)

# Compare with ground-truth labels
nodes_gt, node_sums_gt, times_gt, edges_gt = ground_truth(config['dataset'], config['testing'])
train_nodes_gt, val_nodes_gt, train_times_gt, val_times_gt, train_edge_indexes, val_edge_indexes = \
    train_test_split(
        nodes_gt, 
        torch.stack(times_gt).numpy(),
        edge_indexes,
        test_size=0.2, 
        random_state=seed
        )


# Time ground truth analysis

# Compute Koopman operator
emb_engine, K = get_K(config, train_X)

# Compute eigenvalues and eigenvectors
E, V = np.linalg.eig(K)
idx = np.argsort(np.abs(E))[::-1] # sort eigenvalues and eigenvectors
E = E[idx]
V = V[:, idx]
v = V.real # first two eigenvectors (real parts)

# Compute the Koopman modes
modes = change_basis(states.inputs, v, emb_engine)
val_modes = change_basis(val_X, v, emb_engine)

# Choose eigenvector
mode_idx = config['mode_idx']

r_thr_prec, r_thr_rec, r_thr_f1, r_thr_base = [], [], [], []
r_win_prec, r_win_rec, r_win_f1, r_win_base = [], [], [], []
r_cross, r_corr = [], []
r_mann = []
aucs_nodes = []

for g in tqdm(range(len(val_modes)), desc='Validation dataset', leave=False):

    if val_y[g]==0 or (val_times_gt[g] == 0).all():
        continue

    fig, thr_prec, thr_rec, thr_f1, thr_base = \
        threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                threshold=config['threshold'],
                                window_size=config['window_size'],
                                plot=config['plot'])

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_thr_{mode_idx}.pdf", bbox_inches='tight')
    
    fig, win_prec, win_rec, win_f1, win_base = \
        windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g],
                            window_size=config['window_size'],
                            threshold=config['threshold'],
                            plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_win_{mode_idx}.pdf", bbox_inches='tight')
    
    fig, cc_lag_err, corr = cross_correlation(val_modes[g,:,mode_idx], val_times_gt[g],
                                                plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_cc_{mode_idx}.pdf", bbox_inches='tight')
    
    fig, mw_p_value = mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g], 
                                        window_size=config['window_size'],
                                        plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_mw_{mode_idx}.pdf", bbox_inches='tight')

    node_modes = change_basis(rearrange(val_nodes[g], 't n f -> n t f'), v, emb_engine)
    weights = node_modes[:,-1,mode_idx] - node_modes[:,-1,mode_idx].mean()
    fig, auc = auc_analysis_nodes(np.abs(weights), val_nodes_gt[g][-1], 
                                  val_edge_indexes[g], plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/node_gt/global_dmd/{g}_mask_{mode_idx}.pdf", bbox_inches='tight')
    
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
    aucs_nodes.append(auc)

# Mann-Whitney U test on whole dataset
fig, mw_p_value_dt = mann_whitney_test_dataset(val_modes[val_y==1,:,mode_idx], 
                                                np.stack(val_times_gt)[val_y==1], 
                                                window_size=config['window_size'],
                                                plot=config['plot'])

if fig is not None:
    fig.savefig(f"plots/{config['dataset']}/time_gt/dataset_mw_{mode_idx}.pdf", bbox_inches='tight')

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
    'mw_p_value_dt': mw_p_value_dt,
    'auc_nodes': aucs_nodes
})

# Log on wandb the averages
for key in results.columns:
    wandb.log({f"{key}_avg": np.asarray(results[key]).mean()})

# Save the dataframe to an Excel file in a new sheet
writer = pd.ExcelWriter(path='results.xlsx', engine='xlsxwriter')
results.to_excel(writer, sheet_name=f"time_gt_{config['dataset']}", index=False)


# Spatial ground truth analysis
aucs2, aucs3 = [], []
aucs_nodes = []
for g in tqdm(range(len(edges_gt)), desc='Whole dataset', leave=False):

    if states.targets[g]==0 or torch.sum(edges_gt[g]) == 0:
        continue

    weights = get_weights_from_SINDy(edge_indexes[g], node_states[g], config['dim_red'],
                                     add_self_dependency=config['add_self_dependency_sindy'],
                                     degree=2,
                                     method=config['emb_method'])

    num_nodes = node_states[g].shape[1]
    fig, auc = auc_analysis_edges(weights, edge_indexes[g], edges_gt[g], num_nodes, plot=config['plot'])
    aucs2.append(auc)

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/edge_gt/deg2/{g}_mask.pdf", bbox_inches='tight')
    
    plt.close('all')

    weights = get_weights_from_SINDy(edge_indexes[g], node_states[g], config['dim_red'],
                                     add_self_dependency=config['add_self_dependency_sindy'],
                                     degree=3,
                                     method=config['emb_method'])

    fig, auc = auc_analysis_edges(weights, edge_indexes[g], edges_gt[g], num_nodes, plot=config['plot'])
    aucs3.append(auc)

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/edge_gt/deg3/{g}_mask.pdf", bbox_inches='tight')
    
    plt.close('all')

    # Spatial explanation via DMD
    weights_t = get_weights_from_DMD(node_states[g], 
                                   config['dim_red'],
                                   mode_idx=mode_idx,
                                   method=config['emb_method'])
    weights = weights_t[:,-1] # last time step
    weights = weights - weights.mean()
    fig, auc = auc_analysis_nodes(np.abs(weights), nodes_gt[g][-1], edge_indexes[g], plot=config['plot'])
    aucs_nodes.append(auc)

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/node_gt/local_dmd/{g}_mask.pdf", bbox_inches='tight')
    
    plt.close('all')


results = pd.DataFrame({
    'auc_2': aucs2,
    'auc_3': aucs3,
    'auc_nodes': aucs_nodes
})

# Save the dataframe to an Excel file in a new sheet
results.to_excel(writer, sheet_name=f"edge_gt_{config['dataset']}", index=False)
writer.close()