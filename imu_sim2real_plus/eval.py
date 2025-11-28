import torch
from torch.utils.data import DataLoader, Dataset
from imu_sim2real_plus.models.baseline_net import BaselineIMUOrientation, rot6d_to_quat
from imu_sim2real_plus.metrics.kpi import quat_geodesic_deg, summarize_errors
import numpy as np

class DummyIMUDataset(Dataset):
    def __init__(self, n=20, T=400):
        self.x = torch.randn(n, T, 6)
        self.q = torch.tensor([1,0,0,0]).float().expand(n, T, 4)
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i): return self.x[i], self.q[i]

def main():
    ds = DummyIMUDataset(); dl = DataLoader(ds, batch_size=4)
    model = BaselineIMUOrientation()
    with torch.no_grad():
        all_err = []
        for x, q in dl:
            r6 = model(x)
            qhat = rot6d_to_quat(r6)
            qhat_np = qhat.numpy()
            err = quat_geodesic_deg(q.numpy(), qhat_np).reshape(-1)
            all_err.append(err)
        all_err = np.concatenate(all_err) if all_err else np.array([])
        print(summarize_errors(all_err))

if __name__ == '__main__':
    main()
