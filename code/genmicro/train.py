"""Flow-matching training on the straight path x_t = (1 - t) x0 + t x1, with target velocity v = x1 - x0.

Noise is at t = 0 and data at t = 1; t is drawn from a logit-normal distribution. Hidden convolution
weights are trained with Muon, everything else (embeddings, modulation layers, stem, head, biases) with
AdamW. The learning rate warms up linearly and decays linearly to zero over the final `decay` fraction of
steps. The returned model is an exponential moving average of the weights.
"""
import copy
import os

import torch
import torch.nn.functional as F


def flow_loss(model, x1, cond=None, generator=None):
    """Velocity regression at a logit-normal t."""
    n = len(x1)
    t = torch.sigmoid(torch.randn(n, device=x1.device, generator=generator))
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    tt = t[:, None, None, None]
    v = model((1 - tt) * x0 + tt * x1, t, cond).float()
    return F.mse_loss(v, (x1 - x0).float())


def time_grid(steps, rho=1.0, device="cuda"):
    """Grid from 0 to 1. rho > 1 concentrates steps near t = 1 (fine detail), rho < 1 near t = 0."""
    u = torch.linspace(0, 1, steps + 1, device=device)
    return 1 - (1 - u) ** rho


@torch.no_grad()
def sample(model, x0, steps=16, cond=None, rho=1.0):
    """Heun integration from noise (t=0) to data (t=1); 2*steps - 1 network evaluations."""
    x = x0
    ts = time_grid(steps, rho, x.device)
    for i in range(steps):
        t, dt = ts[i], ts[i + 1] - ts[i]
        v = model(x, t.expand(len(x)), cond)
        xe = x + dt * v
        if i == steps - 1:
            x = xe
        else:
            x = x + dt / 2 * (v + model(xe, ts[i + 1].expand(len(x)), cond))
    return x


torch._dynamo.config.recompile_limit = 64  # one compiled Newton-Schulz kernel per weight shape


def newton_schulz(g, steps: int = 5):
    """Approximate orthogonalisation U V^T of each matrix in a batch (B, m, n).

    Quintic iteration of Jordan et al. (2024), run in bfloat16.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(1) > x.size(2)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return x.mT if tall else x


if os.environ.get("GENMICRO_COMPILE", "1") != "0":
    newton_schulz = torch.compile(newton_schulz, dynamic=False)


class Muon(torch.optim.Optimizer):
    """Nesterov momentum + orthogonalised update for weights viewed as (out, in * kernel).

    Weights with the same shape are orthogonalised together in one batched call.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            ps = [p for p in group["params"] if p.grad is not None]
            bufs = [self.state[p].setdefault("buf", torch.zeros_like(p)) for p in ps]
            grads = [p.grad for p in ps]
            torch._foreach_mul_(bufs, group["momentum"])
            torch._foreach_add_(bufs, grads)
            updates = torch._foreach_add(grads, bufs, alpha=group["momentum"])
            by_shape = {}
            for p, u in zip(ps, updates):
                by_shape.setdefault(p.shape, []).append((p, u))
            for shape, items in by_shape.items():
                g = torch.stack([u.flatten(1) for _, u in items])
                o = newton_schulz(g)
                scale = max(1.0, g.size(1) / g.size(2)) ** 0.5
                params = [p for p, _ in items]
                if group["weight_decay"]:
                    torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(params, [x.view(shape).to(p.dtype) for x, p in zip(o.unbind(0), params)],
                                    alpha=-group["lr"] * scale)


def make_optimizers(model, lr=2e-3, muon_lr=0.02):
    """Muon for 3x3 and 1x1 convolutions inside the network; AdamW for everything else."""
    muon, adam = [], []
    for name, p in model.named_parameters():
        hidden_conv = p.ndim == 4 and not {"stem", "head"} & set(name.split("."))
        (muon if hidden_conv else adam).append(p)
    return [Muon(muon, lr=muon_lr), torch.optim.AdamW(adam, lr=lr, betas=(0.9, 0.99), weight_decay=0.0, fused=True)]


def lr_factor(step, total, warmup, decay):
    if step < warmup:
        return (step + 1) / warmup
    start = int(total * (1 - decay))
    return 1.0 if step < start else max(0.0, (total - step) / (total - start))


class EMA:
    def __init__(self, model, half_life_steps):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.beta = 0.5 ** (1 / max(1, half_life_steps))

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(list(self.model.parameters()), list(model.parameters()), 1 - self.beta)


def train(model, batch_fn, steps, batch=128, lr=2e-3, muon_lr=0.02, warmup=100, decay=0.3, ema_half_life=None,
          seed=0, compile=True, log_every=100):
    """Train `model` in place for `steps` updates and return (EMA model, info).

    `batch_fn(n, step)` returns (x1, cond or None) on the GPU. The EMA half-life defaults to 5% of the steps.
    """
    torch.manual_seed(seed)
    compile = compile and os.environ.get("GENMICRO_COMPILE", "1") != "0"
    opts = make_optimizers(model, lr, muon_lr)
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    ema = EMA(model, ema_half_life or max(1, steps // 20))
    fwd = torch.compile(model) if compile else model
    gen = torch.Generator(device="cuda").manual_seed(seed)
    losses = []
    for step in range(steps):
        x1, cond = batch_fn(batch, step)
        f = lr_factor(step, steps, warmup, decay)
        for o, lrs in zip(opts, base):
            for g, l in zip(o.param_groups, lrs):
                g["lr"] = l * f
        with torch.autocast("cuda", torch.bfloat16):
            loss = flow_loss(fwd, x1, cond, gen)
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        ema.update(model)
        if (step + 1) % 100 == 0:
            losses.append(loss.item())
    return ema.model, dict(loss_every_100_steps=losses)
