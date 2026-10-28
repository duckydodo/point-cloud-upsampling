"""Point cloud upsampling network (PU-GCN/PU-Net style, pure PyTorch) + losses + patch metrics.

Input : (B, N, 3) patch, centred and scaled to the unit sphere.
Output: (B, N*r, 3) upsampled patch.
Design: dense EdgeConv feature extractor -> NodeShuffle-style feature expansion -> coordinate regression
        (residual on the repeated input points) -> one EdgeConv refinement stage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- geometry helpers
def sqdist(a, b):
    """Squared pairwise distances. a (B,M,3), b (B,N,3) -> (B,M,N). Numerically safe for backward."""
    d = (a * a).sum(-1, keepdim=True) + (b * b).sum(-1).unsqueeze(1) - 2.0 * a @ b.transpose(1, 2)
    return d.clamp_min(0.0)


def knn_idx(x, k):
    """Indices of the k nearest neighbours (self included as first). x (B,N,3) -> (B,N,k)."""
    k = min(k, x.shape[1])
    return sqdist(x, x).topk(k, dim=-1, largest=False).indices


def gather_nd(x, idx):
    """x (B,N,C), idx (B,M,k) -> (B,M,k,C)."""
    B, M, k = idx.shape
    base = torch.arange(B, device=x.device).view(B, 1, 1) * x.shape[1]
    flat = (idx + base).reshape(-1)
    return x.reshape(B * x.shape[1], -1)[flat].view(B, M, k, -1)


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(2 * cin, cout)
        self.norm = nn.LayerNorm(cout)

    def forward(self, x, idx):
        nb = gather_nd(x, idx)
        xi = x.unsqueeze(2).expand_as(nb)
        h = F.leaky_relu(self.norm(self.lin(torch.cat([xi, nb - xi], dim=-1))), 0.2)
        return h.max(dim=2).values


class PUNet(nn.Module):
    def __init__(self, up_ratio=4, k=16, width=64, n_layers=3, refine=True):
        super().__init__()
        self.r, self.k, self.refine = up_ratio, k, refine
        self.embed = nn.Linear(3, width)
        self.convs = nn.ModuleList([EdgeConv(width * (i + 1), width) for i in range(n_layers)])
        c = width * (n_layers + 1)
        self.glob = nn.Linear(c, width)
        self.fuse = nn.Sequential(nn.Linear(c + width, 2 * width), nn.LeakyReLU(0.2))
        self.expand = nn.Linear(2 * width, up_ratio * width)          # channel -> point shuffle
        self.head1 = self._head(width)
        if refine:
            self.edge2 = EdgeConv(width, width)
            self.head2 = self._head(width)

    @staticmethod
    def _head(w):
        last = nn.Linear(w, 3)
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, std=1e-3)
        return nn.Sequential(nn.Linear(w, w), nn.LeakyReLU(0.2), last)

    def forward(self, x):
        B, N, _ = x.shape
        idx = knn_idx(x, self.k)
        feats = [self.embed(x)]
        for conv in self.convs:
            feats.append(conv(torch.cat(feats, dim=-1), idx))
        f = torch.cat(feats, dim=-1)
        g = self.glob(f).max(dim=1, keepdim=True).values.expand(-1, N, -1)
        f = self.fuse(torch.cat([f, g], dim=-1))
        up = self.expand(f).reshape(B, N * self.r, -1)                # (B, N*r, width)
        p = x.repeat_interleave(self.r, dim=1) + self.head1(up)
        if self.refine:
            up2 = self.edge2(up, knn_idx(p, self.k))
            p = p + self.head2(up2)
        return p


# ----------------------------------------------------------------------------- losses
def chamfer(pred, gt):
    """Two-sided mean squared Chamfer distance, shape (B,)."""
    d = sqdist(pred, gt)
    return d.min(2).values.mean(1) + d.min(1).values.mean(1)


def repulsion(pred, k=5, radius=0.07, h=0.03):
    """PU-Net repulsion loss (kNN, self excluded), shape (B,)."""
    dk = sqdist(pred, pred).topk(k + 1, dim=-1, largest=False).values[..., 1:]
    dist = dk.clamp_min(1e-12).sqrt()
    return (radius - dist * torch.exp(-dk / h ** 2)).mean(dim=(1, 2))


def uniformity(pred, k=8):
    """Simplified uniformity loss: coefficient of variation of the mean k-NN distance, shape (B,).
    (A lightweight proxy; PU-GAN's disk-based uniform loss is more elaborate.)"""
    dk = sqdist(pred, pred).topk(k + 1, dim=-1, largest=False).values[..., 1:]
    m = dk.clamp_min(1e-12).sqrt().mean(-1)
    return m.std(dim=1) / m.mean(dim=1).clamp_min(1e-8)


def total_loss(pred, gt, cd_scale=1000.0, w_rep=0.1, w_uni=0.1):
    cd = chamfer(pred, gt).mean() * cd_scale
    rep = repulsion(pred).mean() if w_rep > 0 else pred.new_zeros(())
    uni = uniformity(pred).mean() if w_uni > 0 else pred.new_zeros(())
    return cd + w_rep * rep + w_uni * uni, dict(cd=cd.item(), rep=float(rep.detach()), uni=float(uni.detach()))


# ----------------------------------------------------------------------------- metrics + baseline
@torch.no_grad()
def patch_metrics(pred, gt, tau=0.02):
    """Per-patch metrics in normalised units. CD/HD are reported x1e3 (convention of PU-GAN/Grad-PU)."""
    d = sqdist(pred, gt)
    p2g = d.min(2).values                      # (B,M) squared
    g2p = d.min(1).values                      # (B,N) squared
    cd = (p2g.mean(1) + g2p.mean(1)) * 1e3
    hd = torch.maximum(p2g.max(1).values, g2p.max(1).values).sqrt() * 1e3
    prec = (p2g.sqrt() < tau).float().mean(1)
    rec = (g2p.sqrt() < tau).float().mean(1)
    f = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    return dict(cd=cd, hd=hd, f=f, uni=uniformity(pred))


@torch.no_grad()
def interp_patch(x, r):
    """Naive baseline: copy 0 = input point, copy j = midpoint to its j-th nearest neighbour. (B,N,3)->(B,N*r,3)."""
    idx = knn_idx(x, r)                         # (B,N,r) self first
    nb = gather_nd(x, idx)                      # (B,N,r,3)
    out = (x.unsqueeze(2) + nb) / 2.0
    out[:, :, 0] = x
    return out.reshape(x.shape[0], -1, 3)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_upsampler(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=True)
    m = PUNet(up_ratio=ck["up_ratio"], k=ck["k"], width=ck["width"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()
