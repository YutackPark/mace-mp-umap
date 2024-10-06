import os
from typing import List

import tqdm
import torch
import torch.nn as nn

"""
Train structure_encoder from per_atom_features
"""


# keys 
# 'atomic_numbers', 'total_energy', 'atomic_energy', 'inferred_total_energy', 
# 'atomic_features', 'task_id', 'calc_id', 'ionic_step'

# values are not in torch Tensor type


def graph_collate(batch):
    tmp = {}

    batch_indices = []
    per_atom_energy = []  # tagets
    cbi = 0
    for _, datum in enumerate(batch):
        n = len(datum['atomic_numbers'])
        for k, v in datum.items():
            try:
                v = torch.tensor(v)
                if v.dim() == 0:
                    v = v.unsqueeze(-1)
            except TypeError:
                pass
            if k not in tmp:
                tmp[k] = []
            if k == 'atomic_features':
                batch_indices.append(torch.full((v.size(0),), cbi, dtype=torch.long))
                cbi += 1
            if k == 'total_energy':
                per_atom_energy.append(v.clone() / n)
            tmp[k].append(v)

    ret = {}
    for k, v in tmp.items():
        if isinstance(v[0], torch.Tensor):
            ret[k] = torch.cat(v, dim=0)
        else:
            ret[k] = v
    ret["batch"] = torch.cat(batch_indices, dim=0)
    ret["per_atom_energy"] = torch.cat(per_atom_energy, dim=0)

    return ret


class MLP(nn.Module):

    def __init__(
        self, 
        input_size: int, 
        hidden_layers: List[int], 
        output_size: int, 
        use_bias: bool = True,
        activation=nn.SiLU, 
        output_activation=None,
    ):
        super().__init__()
        
        layers = []
        in_features = input_size
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h, bias=use_bias))
            if activation != None:
                layers.append(activation())
            in_features = h
        layers.append(nn.Linear(in_features, output_size, bias=use_bias))
        if output_activation is not None:
            layers.append(output_activation())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class WeightedAverage(nn.Module):
    """
    Learned weighted average (N, M) > (M), N is any integer
    res = Mean(MLP(x) * MLP_w(x), dim='N')
    MLP : N, M -> N, M
    MLP_w : N, M -> N  # output act is sigmoid
    """

    def __init__(
        self,
        hidden_layers: List[int],
        input_size: int = 128,  # for sevennet-0
        use_bias: bool = True,
        activation=nn.SiLU,
    ):
        super().__init__()
        self.input_size = input_size

        self.MLP = MLP(
            input_size, 
            hidden_layers, 
            input_size, 
            use_bias, 
            activation
        )

        self.MLPw = MLP(
            input_size, 
            hidden_layers[:-1], 
            1, use_bias, activation,
            output_activation=nn.Sigmoid,
        )

    def forward(self, x, batch):
        assert self.input_size == x.shape[-1]
        fx = self.MLP(x)
        w = self.MLPw(x)
        weighted = w * fx
        # batch wise sum
        output_size = (batch.max().item() + 1, self.input_size)
        res = torch.zeros(output_size, device=x.device, dtype=x.dtype)
        _batch = batch.unsqueeze(-1).expand_as(weighted)  # broadcast
        res.scatter_reduce_(0, _batch, weighted, reduce='mean')
        return res


class IntrinsicReadout(nn.Module):
    def __init__(
        self,
        hidden_layers: List[int],
        shift: float,
        scale: float,
        input_size: int = 128,  # for sevennet-0
        use_bias: bool = True,
        activation=nn.SiLU,
    ):
        super().__init__()

        self.shift = nn.Parameter(
            torch.FloatTensor([shift]), requires_grad=True,
        )
        self.scale = nn.Parameter(
            torch.FloatTensor([scale]), requires_grad=True,
        )

        self.structure_encode = WeightedAverage(
            hidden_layers, input_size, use_bias, activation
        )
        self.readout = MLP(
            input_size, [input_size // 2], output_size=1, activation=None
        )

    def forward(self, data):
        x = data["atomic_features"]
        batch = data["batch"]

        stct_encode = self.structure_encode(x, batch)
        pred = self.readout(stct_encode) * self.scale + self.shift
        return pred.squeeze(-1), stct_encode

def to_dev(batch, device):
    ret = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            ret[k] = v.to(device)
        else:
            ret[k] = v
    return ret

import glob
import time
from torch.utils.data import DataLoader, random_split
import torch.optim


# input hyper params
batch_size = 128
lr = 0.001  # constant lr, for now
epochs = 300
log = "train.csv"

input_size = 128 # fixed
hidden_nn = [128, 128]
activation = nn.SiLU
use_bias = True

shift = -5.837  # per atom energy mean of chgTot
scale = 1.729   # per atom energy scale of chgTot

criterion = nn.MSELoss()
optim_cls = torch.optim.Adam

print("Don't worry I'm started running", flush=True)
# dataset define
device = torch.device("cuda")

ratio=0.9  # 9:1=train:valid

dataset_f = "../per_atom_vectors/mp_train_*.pth"  # smallest
dataset_files = glob.glob(dataset_f)

dataset = []
for file in dataset_files:
    datalist = torch.load(file)
    print("Reading dataset files...", flush=True)
    dataset.extend(datalist)

train_size = int(ratio * len(dataset))
valid_size = len(dataset) - train_size
train, valid = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=graph_collate)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True, collate_fn=graph_collate)
model = IntrinsicReadout(hidden_nn, shift, scale, use_bias=use_bias, activation=activation)
optimizer = optim_cls(model.parameters(), lr=lr)
model.to(device)

if not os.path.exists(log):
    log_f = open(log, 'w', buffering=1)
    log_f.write("Epoch, Train, Valid, Min\n")
else:
    raise ValueError("log file exist in wd")

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint_0.pt")
print("Start training!!", flush=True)
for epoch in range(epochs):
    start = time.time()
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]")
    train_loss = 0.0
    for batch_idx, batch in pbar:
        batch = to_dev(batch, device)
        # Forward pass
        pred, _ = model(batch)
        loss = criterion(pred, batch["per_atom_energy"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_postfix(Loss=f"{loss.item():.4f}")
    train_loss /= len(train_loader)
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            batch = to_dev(batch, device)
            pred, _ = model(batch)
            loss = criterion(pred, batch["per_atom_energy"])
            valid_loss += loss.item()
    valid_loss /= len(valid_loader)

    duration = (time.time() - start) / 60
    if epoch % 10 == 0:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, f"checkpoint_{epoch}.pt")

    log_f.write(f"{epoch+1}, {train_loss:.4f}, {valid_loss:.4f}, {duration:.2f}\n")

log_f.close()

