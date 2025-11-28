import torch, torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=d*(k-1)//2, dilation=d),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, k, padding=d*(k-1)//2, dilation=d),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class BaselineIMUOrientation(nn.Module):
    def __init__(self, in_ch=6, hid=128):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(in_ch, hid//2, k=5, d=1),
            TCNBlock(hid//2, hid, k=5, d=2),
        )
        self.gru = nn.GRU(hid, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2*hid, 6)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.tcn(x)
        x = x.transpose(1,2)
        y,_ = self.gru(x)
        r6 = self.head(y)
        return r6

def rot6d_to_quat(r6):
    B,T,_ = r6.shape
    a1 = r6[...,:3]; a2 = r6[...,3:]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1*a2).sum(-1, keepdim=True)*b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1,b2,b3], dim=-1)  # (B,T,3,3)
    qw = torch.sqrt(1.0 + R[...,0,0] + R[...,1,1] + R[...,2,2]).clamp_min(1e-6)/2.0
    qx = (R[...,2,1]-R[...,1,2])/(4*qw)
    qy = (R[...,0,2]-R[...,2,0])/(4*qw)
    qz = (R[...,1,0]-R[...,0,1])/(4*qw)
    q = torch.stack([qw,qx,qy,qz], dim=-1)
    q = q / (q.norm(dim=-1, keepdim=True)+1e-8)
    return q
