import torch
import random
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from dataset.utils import (process_classification_dataset,
                            ground_truth)
from utils.utils import get_K, change_basis
from utils.metrics import *


# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

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
        'dim_red': 10,
        'add_self_loops': False, # Too memory-demanding
        'verbose': False,
        'cat_states_layers': True,
        'K_type': 'data', # 'model',
        'testing': False,
        'seed': 42,
        'threshold': None,
        'window_size': 5,
        'plot': True
        }

seed = config['seed']
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


# Load the dataset
dataset, states, node_states, node_labels = process_classification_dataset(config, "DynCRNN", device)
train_X, val_X, train_y, val_y = train_test_split(states.inputs, states.targets, test_size=0.2, random_state=seed)

# Compare with ground-truth labels
nodes_gt, node_sums_gt, times_gt = ground_truth(config['dataset'], config['testing'])
train_node_sums_gt, val_node_sums_gt, train_times_gt, val_times_gt = train_test_split(
    torch.stack(node_sums_gt).numpy(), 
    torch.stack(times_gt).numpy(), 
    test_size=0.2, 
    random_state=seed)

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

columns = ['thr_precision', 'thr_recall', 'thr_f1_score',
           'window_precision', 'window_recall', 'window_f1_score',
           'max_corr_lag_error',
           'mw_p_value',
           'mw_p_value_dt']

r_thr_prec, r_thr_rec, r_thr_f1 = [], [], []
r_win_prec, r_win_rec, r_win_f1 = [], [], []
r_cross = []
r_mann = []

for g in tqdm(range(len(val_modes)), desc='Graphs', leave=False):

    if val_y[g]==0 or (val_times_gt[g] == 0).all():
        continue

    fig, thr_dict = threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                            threshold=config['threshold'],
                                            plot=config['plot'])
    if fig is not None:
        fig.savefig(f'plots/time_gt/{g}_thr_{mode_idx}.png')
    
    fig, win_dict = windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g],
                                        window_size=config['window_size'],
                                        threshold=config['threshold'],
                                        plot=config['plot'])
    if fig is not None:
        fig.savefig(f'plots/time_gt/{g}_win_{mode_idx}.png')
    
    fig, cc_dict = cross_correlation(val_modes[g,:,mode_idx], val_times_gt[g],
                                     plot=config['plot'])
    if fig is not None:
        fig.savefig(f'plots/time_gt/{g}_cc_{mode_idx}.png')
    
    fig, mw_dict = mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g], 
                                    window_size=config['window_size'],
                                    plot=config['plot'])
    if fig is not None:
        fig.savefig(f'plots/time_gt/{g}_mw_{mode_idx}.png')
    
    plt.close('all')
    
    thr_prec, thr_rec, thr_f1 = thr_dict[columns[0]], thr_dict[columns[1]], thr_dict[columns[2]]
    win_prec, win_rec, win_f1 = win_dict[columns[3]], win_dict[columns[4]], win_dict[columns[5]]
    cc_lag = cc_dict[columns[6]]
    mw_p = mw_dict[columns[7]]
    
    r_thr_prec.append(thr_prec)
    r_thr_rec.append(thr_rec)
    r_thr_f1.append(thr_f1)
    r_win_prec.append(win_prec)
    r_win_rec.append(win_rec)
    r_win_f1.append(win_f1)
    r_cross.append(cc_lag)
    r_mann.append(mw_p)

# Mann-Whitney U test on whole dataset
fig, mw_p_value_dt = mann_whitney_test_dataset(val_modes[val_y==1,:,mode_idx], 
                                                    np.stack(val_times_gt)[val_y==1], 
                                                    window_size=config['window_size'],
                                                    plot=config['plot'])
if fig is not None:
    fig.savefig(f'plots/time_gt/dataset_mw_{mode_idx}.png')

# Create a dataframe with the results
results = pd.DataFrame({
    'thr_precision': r_thr_prec,
    'thr_recall': r_thr_rec,
    'thr_f1_score': r_thr_f1,
    'window_precision': r_win_prec,
    'window_recall': r_win_rec,
    'window_f1_score': r_win_f1,
    'max_corr_lag_error': r_cross,
    'mw_p_value': r_mann,
    'mw_p_value_dt': mw_p_value_dt[columns[8]]
})

# Save the dataframe to a CSV file in a new sheet
results.to_csv('results.csv', header=columns, index=False)

