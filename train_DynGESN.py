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
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.MSELoss()(output, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.MSELoss()(output, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.MSELoss()(output, y)
        self.log('test_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.07)
        return optimizer

forecaster = LinearRegression(input_size=config.reservoir_size*config.num_layers, output_size=1).to(device)

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_loss',
    mode='min',
)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=5, verbose=True, mode="min")

wandb_logger = WandbLogger(name='dyngesn',project='koopman')

trainer = pl.Trainer(max_epochs=config.epochs,
                    logger=wandb_logger,
                    devices=1, 
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    # limit_train_batches=0.1, 
                    # limit_val_batches=0.1,
                    # limit_train_batches=100,  # end an epoch after 100 updates
                    callbacks=[checkpoint_callback, early_stop_callback],
                    deterministic=True)

trainer.fit(model=forecaster, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)