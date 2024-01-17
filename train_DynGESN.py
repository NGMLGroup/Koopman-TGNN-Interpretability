import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import random
import numpy as np

from tsl.metrics.torch import MaskedMAE
from tsl.engines import Predictor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange

from dataset.utils import process_PVUS

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
pl.seed_everything(42)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = {
    'reservoir_size': 100,
    'input_scaling': 1.,
    'reservoir_layers': 1,
    'leaking_rate': 0.9,
    'spectral_radius': 0.9,
    'density': 0.5,
    'reservoir_activation': 'tanh',
    'alpha_decay': False,
    'epochs': 100,
    'lr': 0.001
}

train_dataloader, test_dataloader, val_dataloader = process_PVUS(config, device, ignore_file=True, verbose=False)

wandb.init(project="koopman", config=config)
config = wandb.config

class LinearRegression(pl.LightningModule):
    def __init__(self, encoder, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.encoder = encoder
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, edge_index, edge_weight):
        z = self.encoder(x, edge_index, edge_weight)
        b, n, f = z.shape
        new_x = self.linear(z)
        new_x = rearrange(new_x, 'b n f -> f n b', n=n, f=self.output_size)

        return new_x

model_file_path = "models/saved/DynGESN.pt"
model = torch.load(model_file_path)
forecaster = LinearRegression(model, input_size=config.reservoir_size*config.num_layers, output_size=feat_size*horizon).to(device)

loss_fn = MaskedMAE()

metrics = {'mae': MaskedMAE()}

# setup predictor
predictor = Predictor(
    model=forecaster,              # our initialized model
    optim_class=torch.optim.Adam,  # specify optimizer to be used...
    optim_kwargs={'lr': config.lr},    # ...and parameters for its initialization
    loss_fn=loss_fn,               # which loss function to be used
    metrics=metrics                # metrics to be logged during train/val/test
)

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)
early_stop_callback = EarlyStopping(monitor="val_mae", min_delta=0.01, patience=3, verbose=False, mode="max")

wandb_logger = WandbLogger(name='dyngesn',project='koopman')

trainer = pl.Trainer(max_epochs=config.epochs,
                     logger=wandb_logger,
                     devices=1, 
                     accelerator="gpu" if torch.cuda.is_available() else "cpu",
                     limit_train_batches=100,  # end an epoch after 100 updates
                     callbacks=[checkpoint_callback, early_stop_callback],
                     deterministic=True)

trainer.fit(predictor, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)