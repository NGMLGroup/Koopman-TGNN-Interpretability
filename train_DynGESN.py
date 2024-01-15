import torch
import wandb
import torch.nn as nn
import pytorch_lightning as pl
import random
import tsl
import numpy as np

from tsl.datasets import PvUS
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.metrics.torch import MaskedMAE
from tsl.engines import Predictor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange

from models.DynGraphESN import DynGESNModel

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
pl.seed_everything(42)

config = {
    'window': 128,
    'stride': 1,
    'reservoir_size': 100,
    'input_scaling': 1.,
    'num_layers': 1,
    'leaking_rate': 0.9,
    'spectral_radius': 0.9,
    'density': 0.5,
    'activation': 'tanh',
    'epochs': 100,
    'lr': 0.001
}

wandb.init(project="koopman", config=config)
config = wandb.config

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataset = PvUS(root="dataset", zones=['west'])
sim = dataset.get_similarity("distance")
connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")

horizon = 24
torch_dataset = tsl.data.SpatioTemporalDataset(target=dataset.dataframe(),
                                      connectivity=connectivity,
                                      mask=dataset.mask,
                                      horizon=horizon,
                                      window=config.window,
                                      stride=config.stride)

sample = torch_dataset[0].to(device)
time_interval, num_nodes, feat_size = sample.input.x.shape

scalers = {'target': StandardScaler(axis=(0, 1))}
splitter = TemporalSplitter(val_len=0.1, test_len=0.2)
dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=1,
)
dm.setup()

model = DynGESNModel(input_size=feat_size,
                reservoir_size=config.reservoir_size,
                input_scaling=config.input_scaling,
                reservoir_layers=config.num_layers,
                leaking_rate=config.leaking_rate,
                spectral_radius=config.spectral_radius,
                density=config.density,
                reservoir_activation=config.activation,
                alpha_decay=False).to(device)

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
early_stop_callback = EarlyStopping(monitor="val_mae", min_delta=0.0, patience=5, verbose=True, mode="min")

wandb_logger = WandbLogger(name='dyngesn',project='koopman')

trainer = pl.Trainer(max_epochs=config.epochs,
                    logger=wandb_logger,
                    devices=1, 
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    limit_train_batches=0.1, 
                    limit_val_batches=0.1,
                    #  limit_train_batches=100,  # end an epoch after 100 updates
                    callbacks=[checkpoint_callback, early_stop_callback],
                    deterministic=True)

trainer.fit(predictor, datamodule=dm)