"""Velocity network for flow matching on micrographs: a fully convolutional U-Net.

* No attention and no positional input, so the network is translation-equivariant and generates images
  of any size (a multiple of 8 px with four levels).
* Residual blocks with channel RMS normalisation and adaLN-zero modulation: the flow time sets a
  per-channel scale, shift and gate; gates start at zero, so every block starts as the identity.
* Lossless resampling (pixel-unshuffle down, pixel-shuffle up) with 1x1 projections; an optional
  patchify stem (patch > 1) runs all convolutions at reduced resolution.
* Extra image channels (`image_cond`), such as an upsampled layout, are concatenated to the input.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional wider default; the processing experiments explicitly pass their narrower configuration
# from study.MODEL, and checkpoint configurations are restored without substituting this default.
UNET = dict(patch=1, widths=(64, 128, 256, 256), depths=(1, 2, 2, 4))


def rms_norm(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(1, keepdim=True) + eps)


class Fourier(nn.Module):
    """Fixed sinusoidal features of a scalar in [0, 1]."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.linspace(0, 10, dim // 2), persistent=False)

    def forward(self, t):
        a = t[:, None].float() * self.freq
        return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):
    def __init__(self, width, emb_dim, periodic=False):
        super().__init__()
        self.mod = nn.Linear(emb_dim, 3 * width)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, padding_mode="circular" if periodic else "zeros")
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, padding_mode="circular" if periodic else "zeros")

    def forward(self, x, emb):
        scale, shift, gate = self.mod(emb)[:, :, None, None].chunk(3, 1)
        h = rms_norm(x) * (1 + scale) + shift
        h = self.conv2(F.silu(self.conv1(F.silu(h))))
        return x + gate * h


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Conv2d(4 * cin, cout, 1)

    def forward(self, x):
        return self.proj(F.pixel_unshuffle(x, 2))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.proj = nn.Conv2d(cin, 4 * cout, 1)

    def forward(self, x):
        return F.pixel_shuffle(self.proj(x), 2)


class FlowUNet(nn.Module):
    def __init__(self, channels=1, widths=(128, 256, 256), depths=(2, 2, 4), patch=2, emb_dim=256, image_cond=0, cond_dim=0, periodic=False):
        super().__init__()
        self.patch = patch
        self.cond = ConditionEmbedding(cond_dim, emb_dim) if cond_dim else None
        self.time = nn.Sequential(Fourier(128), nn.Linear(128, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.stem = nn.Conv2d((channels + image_cond) * patch**2, widths[0], 3, padding=1, padding_mode="circular" if periodic else "zeros")
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i, (w, d) in enumerate(zip(widths, depths)):
            last = i == len(widths) - 1
            # The deepest level is the bottleneck: its blocks run once, in the encoder.
            self.enc.append(nn.ModuleList(Block(w, emb_dim, periodic) for _ in range(d)))
            self.dec.append(nn.ModuleList([] if last else [Block(w, emb_dim, periodic) for _ in range(d)]))
            if not last:
                self.down.append(Down(w, widths[i + 1]))
                self.up.append(Up(widths[i + 1], w))
        self.head = nn.Conv2d(widths[0], channels * patch**2, 3, padding=1, padding_mode="circular" if periodic else "zeros")
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t, cond=None, image=None):
        """x (B, channels, H, W) at flow time t (B,); image (B, image_cond, H, W) or None."""
        if image is not None:
            x = torch.cat([x, image], 1)
        emb = F.silu(self.time(t) + (self.cond(cond) if self.cond is not None and cond is not None else 0))
        h = self.stem(F.pixel_unshuffle(x, self.patch))
        skips = []
        for i, blocks in enumerate(self.enc):
            for b in blocks:
                h = b(h, emb)
            if i < len(self.down):
                skips.append(h)
                h = self.down[i](h)
        for i in reversed(range(len(self.up))):
            h = self.up[i](h) + skips.pop()
            for b in self.dec[i]:
                h = b(h, emb)
        return F.pixel_shuffle(self.head(F.silu(rms_norm(h))), self.patch)


class ConditionEmbedding(nn.Module):
    """Scaled coordinates plus fixed sinusoidal features, including the conditioning flag."""
    def __init__(self, cond_dim, width, n_freq=16):
        super().__init__()
        self.register_buffer("freq", math.pi * 2.0 ** torch.linspace(-1,4,n_freq), persistent=False)
        self.mlp=nn.Sequential(nn.Linear(cond_dim*(1+2*n_freq),width),nn.SiLU(),nn.Linear(width,width))
    def forward(self,c):
        if self.freq.numel():
            a=c[:,:,None].float()*self.freq
            c=torch.cat([c,a.sin().flatten(1),a.cos().flatten(1)],1)
        return self.mlp(c)
