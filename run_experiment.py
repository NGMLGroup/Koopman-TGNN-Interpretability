import subprocess
import argparse
import wandb
import json


# Initialize wandb
wandb.init(project='Koopman')
run_id = wandb.run.id


parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('--encoder_type', type=str, default='dyngcrnn', help='Type of encoder')
parser.add_argument('--dataset', type=str, default='infectious_ct1', help='Name of the dataset')
parser.add_argument('--emb_method', type=str, default='PCA', help='Embedding method')
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='Verbose')
parser.add_argument('--beta', type=float, default=0.1, help='Weight of the ridge loss')
parser.add_argument('--seed', type=int, default=42, help='Seed')
parser.add_argument('--sweep', action=argparse.BooleanOptionalAction, help='Sweep')

args = parser.parse_args()

# Load configuration from JSON file
encoder_type = args.encoder_type
dataset_name = args.dataset
config_file = 'configs/GCRN_config.json'
with open(config_file, 'r') as f:
    configs = json.load(f)

# Retrieve the configuration for the selected model
if encoder_type not in configs:
    raise ValueError(f"Hyperparameters for encoder {encoder_type} are missing.")
configs = configs[encoder_type]

if dataset_name not in configs:
    raise ValueError(f"Hyperparameters for dataset {dataset_name} are missing.")

# Overwrite json configs with args
config = configs[dataset_name]
for key, value in vars(args).items():
    config[key] = value

# Log the configuration to wandb
wandb.config.update(config)

# Construct the command to run train_tsl_model.py
train_command = [
    'python', 'train_tsl_model.py',
    '--dataset', args.dataset,
    '--hidden_size', str(config['hidden_size']),
    '--rnn_layers', str(config['rnn_layers']),
    '--readout_layers', str(config['readout_layers']),
    '--dim_red', str(config['dim_red']),
    '--weight_decay', str(config['weight_decay']),
    '--step_size', str(config['step_size']),
    '--gamma', str(config['gamma']),
    '--encoder_type', str(config['encoder_type']),
    '--k_kernel', str(config['k_kernel']),
    '--beta', str(config['beta']),
    '--batch_size', str(config['batch_size']),
    '--max_epochs', str(config['max_epochs']),
    '--patience', str(config['patience']),
    '--min_delta', str(config['min_delta']),
    '--seed', str(config['seed']),
    '--wandb_run_id', run_id
]

# Construct the command to run experiment_graph.py
experiment_command = [
    'python', 'experiment_graph.py',
    '--dataset', args.dataset,
    '--encoder_type', str(config['encoder_type']),
    '--emb_method', str(config['emb_method']),
    '--seed', str(config['seed']),
    '--mode_idx', str(config['mode_idx']),
    '--threshold', str(config['threshold']),
    '--window_size', str(config['window_size']),
    '--wandb_run_id', str(run_id)
]

# Add boolean flags
if config['verbose']:
    train_command.append('--verbose')
else:
    train_command.append('--no-verbose')
if config['sweep']:
    train_command.append('--sweep')
    experiment_command.append('--sweep')
else:
    train_command.append('--no-sweep')
    experiment_command.append('--no-sweep')
if config['plot']:
    experiment_command.append('--plot')
else:
    experiment_command.append('--no-plot')
if config['cat_states_layers']:
    train_command.append('--cat_states_layers')
else:
    train_command.append('--no-cat_states_layers')
if config['self_loop']:
    train_command.append('--self_loop')
    experiment_command.append('--add_self_loops')
else:
    train_command.append('--no-self_loop')
    experiment_command.append('--no-add_self_loops')
if config['add_self_dependency_sindy']:
    experiment_command.append('--add_self_dependency_sindy')
else:
    experiment_command.append('--no-add_self_dependency_sindy')
if config['testing']:
    experiment_command.append('--testing')
else:
    experiment_command.append('--no-testing')

# Execute the training script
print('Training the model...')
subprocess.run(train_command, check=True)

# Execute the experiment script
print('Running the experiment...')
subprocess.run(experiment_command, check=True)