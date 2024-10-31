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
                         get_weights_from_DMD,
                         get_weights_from_PCA,
                         run_saliency)
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
indices = list(range(len(node_labels)))
train_X, val_X, train_y, val_y, train_nodes, val_nodes, train_idx, val_idx = \
    train_test_split(states.inputs, 
                     states.targets, 
                     node_states,
                     indices,
                     test_size=0.2, random_state=seed)
edge_indexes, node_labels, graph_labels = load_classification_dataset(config['dataset'], False)

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


# Explanation on validation dataset

## Compute the Koopman modes
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

# Compute saliency weights as baseline
sal_attr = run_saliency(edge_indexes, node_labels, graph_labels, 
                        config, device, verbose=False)

# Compute the PCA weights as baseline
v_id = np.eye(v.shape[0])
pca_modes = change_basis(states.inputs, v_id, emb_engine)
val_pca_modes = change_basis(val_X, v_id, emb_engine)

gs = []
r_thr_prec, r_thr_rec, r_thr_f1, r_thr_base = [], [], [], []
r_mad_prec, r_mad_rec, r_mad_f1, r_mad_base = [], [], [], []
r_win_prec, r_win_rec, r_win_f1, r_win_base = [], [], [], []
r_sal_prec, r_sal_rec, r_sal_f1 = [], [], []
r_thr_pca_base, r_win_pca_base = [], []
r_mann = []
aucs_nodes, aucs_nodes_sal_val, aucs_nodes_pca_val = [], [], []

for g in tqdm(range(len(val_modes)), desc='Validation dataset', leave=False):

    if val_y[g]==0 or (val_times_gt[g] == 0).all():
        continue

    gs.append(g)

    fig, thr_prec, thr_rec, thr_f1, thr_base = \
        threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                threshold=config['threshold'],
                                window_size=config['window_size'],
                                plot=config['plot'])

    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_thr_{mode_idx}.pdf", bbox_inches='tight')

    fig, mad_prec, mad_rec, mad_f1, mad_base = \
        threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                threshold='mad',
                                window_size=config['window_size'],
                                plot=config['plot'])
    
    # Baseline with PCA modes
    _, _, _, thr_pca_base, _ = \
        threshold_based_detection(val_pca_modes[g,:,0], val_times_gt[g], # first PC
                                threshold=config['threshold'],
                                window_size=config['window_size'],
                                plot=False)
    
    fig, win_prec, win_rec, win_f1, win_base = \
        windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g],
                            window_size=config['window_size'],
                            threshold=config['threshold'],
                            plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_win_{mode_idx}.pdf", bbox_inches='tight')

    # Baseline with PCA modes
    _, _, _, win_pca_base, _ = \
        windowing_analysis(val_pca_modes[g,:,0], val_times_gt[g], # first PC
                            window_size=config['window_size'],
                            threshold=config['threshold'],
                            plot=False)

    # Baseline with saliency map
    fig, sal_prec, sal_rec, sal_f1 = F1_baseline_saliency(sal_attr[val_idx[g]].cpu().numpy().sum(axis=-1),
                                                          val_times_gt[g],
                                                          window_size=config['window_size'],
                                                          plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_sal_{mode_idx}.pdf", bbox_inches='tight')
    
    fig, mw_p_value = mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g], 
                                        window_size=config['window_size'],
                                        plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/time_gt/{g}_mw_{mode_idx}.pdf", bbox_inches='tight')

    # Spatial explanation on nodes via DMD on modes from trianing dataset
    node_modes = change_basis(rearrange(val_nodes[g], 't n f -> n t f'), v, emb_engine)
    weights = node_modes[:,-1,mode_idx] - node_modes[:,-1,mode_idx].mean()
    fig, auc = auc_analysis_nodes(np.abs(weights), val_nodes_gt[g][-1], 
                                  val_edge_indexes[g], plot=config['plot'])
    if fig is not None:
        fig.savefig(f"plots/{config['dataset']}/node_gt/global_dmd/{g}_mask_{mode_idx}.pdf", bbox_inches='tight')
    
    # Spatial explanation on nodes via saliency
    weights_t = sal_attr[val_idx[g]].T.cpu().numpy() # shape [nodes, times]
    weights = np.max(np.abs(weights_t), axis=1) # max over time
    _, auc_sal = auc_analysis_nodes(weights, val_nodes_gt[g][-1], 
                                  val_edge_indexes[g], plot=False)

    # Spatial explanation on nodes via PCA only
    node_modes = change_basis(rearrange(val_nodes[g], 't n f -> n t f'), v_id, emb_engine)
    weights = node_modes[:,-1,0] - node_modes[:,-1,0].mean() # first PC
    _, auc_pca = auc_analysis_nodes(np.abs(weights), val_nodes_gt[g][-1], 
                                  val_edge_indexes[g], plot=False)
    
    plt.close('all')
    
    r_thr_prec.append(thr_prec)
    r_thr_rec.append(thr_rec)
    r_thr_f1.append(thr_f1)
    r_thr_base.append(thr_base)
    r_mad_prec.append(mad_prec)
    r_mad_rec.append(mad_rec)
    r_mad_f1.append(mad_f1)
    r_mad_base.append(mad_base)
    r_thr_pca_base.append(thr_pca_base)
    r_win_prec.append(win_prec)
    r_win_rec.append(win_rec)
    r_win_f1.append(win_f1)
    r_win_base.append(win_base)
    r_win_pca_base.append(win_pca_base)
    r_sal_prec.append(sal_prec)
    r_sal_rec.append(sal_rec)
    r_sal_f1.append(sal_f1)
    r_mann.append(mw_p_value)
    aucs_nodes.append(auc)
    aucs_nodes_sal_val.append(auc_sal)
    aucs_nodes_pca_val.append(auc_pca)

# Mann-Whitney U test on whole dataset
fig, mw_p_value_dt = mann_whitney_test_dataset(val_modes[val_y==1,:,mode_idx], 
                                                np.stack(val_times_gt)[val_y==1], 
                                                window_size=config['window_size'],
                                                plot=config['plot'])

if fig is not None:
    fig.savefig(f"plots/{config['dataset']}/time_gt/dataset_mw_{mode_idx}.pdf", bbox_inches='tight')

# Create a dataframe with the results
results = pd.DataFrame({
    'g': gs,
    'thr_precision': r_thr_prec,
    'thr_recall': r_thr_rec,
    'thr_f1_score': r_thr_f1,
    'thr_baseline_f1': r_thr_base,
    'mad_precision': r_mad_prec,
    'mad_recall': r_mad_rec,
    'mad_f1_score': r_mad_f1,
    'mad_baseline_f1': r_mad_base,
    'thr_pca_baseline_f1': r_thr_pca_base,
    'window_precision': r_win_prec,
    'window_recall': r_win_rec,
    'window_f1_score': r_win_f1,
    'window_baseline_f1': r_win_base,
    'window_pca_baseline_f1': r_win_pca_base,
    'saliency_precision': r_sal_prec,
    'saliency_recall': r_sal_rec,
    'saliency_f1_score': r_sal_f1,
    'mw_p_value': r_mann,
    'mw_p_value_dt': mw_p_value_dt,
    'auc_nodes': aucs_nodes,
    'auc_nodes_sal_val': aucs_nodes_sal_val,
    'auc_nodes_pca_val': aucs_nodes_pca_val
})

# Log on wandb the averages
for key in results.columns:
    wandb.log({f"{key}_avg": np.asarray(results[key]).mean()})

# Save the dataframe to an Excel file in a new sheet
writer = pd.ExcelWriter(path='results.xlsx', engine='xlsxwriter')
results.to_excel(writer, sheet_name=f"time_gt_{config['dataset']}", index=False)


# Explanation on whole dataset
gs = []
aucs2, aucs3 = [], []
aucs_nodes = []
aucs_nodes_sal_base, aucs_nodes_pca_base = [], []

for g in tqdm(range(len(edges_gt)), desc='Whole dataset', leave=False):

    if states.targets[g]==0 or torch.sum(edges_gt[g]) == 0:
        continue

    gs.append(g)

    # Spatial explanation via SINDy on edges
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

    # Spatial explanation on nodes via DMD on graph modes
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

    # Spatial explanation on nodes baseline via saliency
    weights_t = sal_attr[g].T.cpu().numpy() # shape [nodes, times]
    weights = np.max(np.abs(weights_t), axis=1) # max over time
    _, auc = auc_analysis_nodes(weights, nodes_gt[g][-1], edge_indexes[g], plot=config['plot'])
    aucs_nodes_sal_base.append(auc)

    # Spatial explanation on nodes baseline via PCA only
    weights_t = get_weights_from_PCA(node_states[g], config['dim_red'], method=config['emb_method'])
    weights = weights_t[:,-1,0] # last time step, first PC
    weights = weights - weights.mean()
    _, auc = auc_analysis_nodes(np.abs(weights), nodes_gt[g][-1], edge_indexes[g], plot=False)
    aucs_nodes_pca_base.append(auc)
    
    plt.close('all')


results = pd.DataFrame({
    'g': gs,
    'auc_2': aucs2,
    'auc_3': aucs3,
    'auc_nodes': aucs_nodes,
    'auc_nodes_pca_base': aucs_nodes_pca_base,
    'auc_nodes_sal_base': aucs_nodes_sal_base
})

# Save the dataframe to an Excel file in a new sheet
results.to_excel(writer, sheet_name=f"edge_gt_{config['dataset']}", index=False)
writer.close()