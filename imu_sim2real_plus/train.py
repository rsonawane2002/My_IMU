import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from imu_sim2real_plus.models.baseline_net import BaselineIMUOrientation, rot6d_to_quat

class DummyIMUDataset(Dataset):
    def __init__(self, n=100, T=400):
        self.x = torch.randn(n, T, 6)
        self.q = torch.tensor([1,0,0,0]).float().expand(n, T, 4)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.q[i]

def train_epoch(model, loader, opt):
    model.train()
    loss_sum = 0.0
    for x, q in loader:
        opt.zero_grad()
        r6 = model(x)
        qhat = rot6d_to_quat(r6)
        dot = torch.abs((qhat*q).sum(-1))
        loss = (1.0 - dot).mean()
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
    return loss_sum/len(loader.dataset)

def main():
    ds = DummyIMUDataset(); dl = DataLoader(ds, batch_size=8, shuffle=True)
    model = BaselineIMUOrientation()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(3):
        l = train_epoch(model, dl, opt)
        print(f'epoch {e}: loss={l:.4f}')

if __name__ == '__main__':
    main()
