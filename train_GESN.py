import torch
import wandb
import os
import random
import numpy as np

from torch_geometric.datasets import TUDataset
from tqdm import trange
from models.GraphESN import GraphESN
from sklearn.model_selection import train_test_split

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

config = {
    'input_size': 7,
    'hidden_size': 286,
    'conv_steps': 52,
    'input_scaling': 0.26371404282791244,
    'num_layers': 1,
    'leaking_rate': 0.5140825833176746,
    'spectral_radius': 3.1610006979082925,
    'density': 0.8557646803055511,
    'activation': 'self_norm',
    'epochs': 462,
    'lr': 0.08292410152754211
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


model = GraphESN(input_size=config.input_size,
                hidden_size=config.hidden_size,
                steps=config.conv_steps,
                input_scaling=config.input_scaling,
                num_layers=config.num_layers,
                leaking_rate=config.leaking_rate,
                spectral_radius=config.spectral_radius,
                density=config.density,
                activation=config.activation,
                alpha_decay=False)

embs = []
for g in dataset:
    embs.append(model(g['x'], g['edge_index']).mean(dim=0))
embs = torch.stack(embs).to(device)

X_train, X_test, y_train, y_test = train_test_split(embs, dataset.y, test_size=0.3, random_state=seed)

classifier = torch.nn.Sequential(
    torch.nn.Linear(model.hidden_size, 2)
    ).to(device)

optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)
loss_fn = torch.nn.CrossEntropyLoss()

classifier.train()
losses, loss_e = [], []

for epoch in trange(config.epochs):
    for x, y in zip(X_train, y_train):
        out = classifier(x)
        optimizer.zero_grad()
        loss = loss_fn(out, y.type(torch.LongTensor).squeeze().to(device))
        wandb.log({'loss': loss, 'epoch': epoch})
        loss.backward()
        optimizer.step()
        loss_e.append(loss.clone().detach())
    losses.append(torch.tensor(loss_e).mean())

classifier.eval()
test_acc = (classifier(X_test).cpu().argmax(dim=-1) == y_test).sum()/len(y_test)
print(f"Test accuracy: {test_acc}")

wandb.log({'test_acc': test_acc})