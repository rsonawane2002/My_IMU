
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from imu_sim2real_plus.models.baseline_net import BaselineIMUOrientation, rot6d_to_quat
import numpy as np
import os

class StationaryIMUDataset(Dataset):
    def __init__(self, data_dir):
        self.f_meas = np.load(os.path.join(data_dir, 'seq_00000_f_meas.npy'))
        self.w_meas = np.load(os.path.join(data_dir, 'seq_00000_w_meas.npy'))
        self.x = torch.from_numpy(np.concatenate([self.f_meas, self.w_meas], axis=1)).float()
        self.q = torch.tensor([1,0,0,0]).float().expand(self.x.shape[0], 4)

    def __len__(self): 
        return 1 # Only one sequence

    def __getitem__(self, i):
        return self.x, self.q

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
    ds = StationaryIMUDataset('runs/stationary_no_vibration'); dl = DataLoader(ds, batch_size=1, shuffle=True)
    model = BaselineIMUOrientation()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(10):
        l = train_epoch(model, dl, opt)
        print(f'epoch {e}: loss={l:.4f}')
    
    torch.save(model.state_dict(), 'runs/stationary_model.pth')

if __name__ == '__main__':
    main()
