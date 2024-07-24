import torch
import random
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

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
           'mw_p_value']

for g in tqdm(range(len(val_modes)), desc='Graphs', leave=False):

    if val_y[g]==0 or (val_times_gt[g] == 0).all():
        continue

    fig, thr_dict = threshold_based_detection(val_modes[g,:,mode_idx], val_times_gt[g], 
                                            threshold=config['threshold'],
                                            plot=True)
    fig.savefig(f'plots/time_gt/{g}_thr_{mode_idx}.png')
    
    fig, win_dict = windowing_analysis(val_modes[g,:,mode_idx], val_times_gt[g],
                                        window_size=config['window_size'],
                                        threshold=config['threshold'],
                                        plot=True)
    fig.savefig(f'plots/time_gt/{g}_win_{mode_idx}.png')
    
    fig, cc_dict = cross_correlation(val_modes[g,:,mode_idx], val_times_gt[g],
                                     plot=True)
    fig.savefig(f'plots/time_gt/{g}_cc_{mode_idx}.png')
    
    fig, mw_dict = mann_whitney_test(val_modes[g,:,mode_idx], val_times_gt[g], 
                                    window_size=config['window_size'],
                                    plot=True)
    fig.savefig(f'plots/time_gt/{g}_mw_{mode_idx}.png')

    plt.close('all')

