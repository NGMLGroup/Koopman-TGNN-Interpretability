import torch
import wandb
import os
import random
import numpy as np

from torch_geometric.datasets import TUDataset
from tqdm import trange
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

config = {
    'lr': 0.08292410152754211,
    'n_layers': 3,
    'activation': 'tanh',
    'h_dim': 32,
    'epochs': 100
}

wandb.init(project="koopman", config=config)
config = wandb.config

# Select one GPU if more are available
os.environ["CUDA_VISIBLE_DEVICES"]='0'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


dataset = TUDataset(root='/ESGNN/data/MUTAG', name='MUTAG')

koops = torch.load('KOP/quadr_TruncatedSVD_5.pt')

K_train, K_test, y_train, y_test = train_test_split(koops, dataset.y, test_size=0.3, random_state=seed)

modules = []
if config.activation == 'relu':
    activation = torch.nn.ReLU()
elif config.activation == 'tanh':
    activation = torch.nn.Tanh()

for i in range(config.n_layers):
    in_dim = koops.shape[1] if i == 0 else config.h_dim
    out_dim = 2 if i == config.n_layers-1 else config.h_dim
    modules.append(torch.nn.Linear(in_dim, out_dim))
    if i != config.n_layers-1:
        modules.append(activation)

classifier = torch.nn.Sequential(*modules).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = config.epochs
classifier.train()
losses, loss_e = [], []

for epoch in trange(epochs):
    for k, y in zip(K_train, y_train):
        out = classifier(k)
        optimizer.zero_grad()
        loss = loss_fn(out, y.type(torch.LongTensor).squeeze().to(device))
        wandb.log({'loss': loss, 'epoch': epoch})
        loss.backward()
        optimizer.step()
        loss_e.append(loss.clone().detach())
    losses.append(torch.tensor(loss_e).mean())

classifier.eval()
test_acc = (classifier(K_test).cpu().argmax(dim=-1) == y_test).sum()/len(y_test)
print(f"Test accuracy: {test_acc}")

wandb.log({'test_acc': test_acc})