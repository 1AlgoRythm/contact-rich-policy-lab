"""Three small policies: MLP behaviour cloning, transformer action chunking, 1D conditional diffusion policy."""
import math

import numpy as np
import torch
import torch.nn as nn

from .diffusion import Schedule, ddim_sample

ACT_DIM = 3
K_CHUNK = 8  # chunk length for the transformer and the diffusion policy
N_EXEC = 4  # actions executed from each chunk before re-planning


class MLPPolicy(nn.Module):
    """Obs -> the next action (chunk length 1)."""
    K = 1

    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ACT_DIM))

    def forward(self, obs):  # (B, D) -> (B, 1, 3)
        return self.net(obs)[:, None, :]

    def loss(self, obs, chunk):  # chunk: (B, K, 3); only the first action is used
        return ((self(obs) - chunk[:, :1]) ** 2).mean()


class TransformerChunkPolicy(nn.Module):
    """Obs -> K actions at once. One observation token plus K learned query tokens go through a transformer encoder
    (self-attention lets every future-action query look at the observation and at the other queries); each query's
    output is projected to one 3-D action. Action chunking as in ACT, WITHOUT the CVAE latent."""
    K = K_CHUNK

    def __init__(self, obs_dim, d=96, heads=4, layers=2, ff=192):
        super().__init__()
        self.obs_in = nn.Linear(obs_dim, d)
        self.queries = nn.Parameter(torch.randn(K_CHUNK, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ACT_DIM))

    def forward(self, obs):  # (B, D) -> (B, K, 3)
        B = obs.shape[0]
        tokens = torch.cat([self.obs_in(obs)[:, None, :], self.queries[None].expand(B, -1, -1)], dim=1)  # (B, 1+K, d)
        return self.out(self.enc(tokens)[:, 1:])

    def loss(self, obs, chunk):
        return ((self(obs) - chunk) ** 2).mean()


def timestep_embedding(t, dim=64):
    """Sinusoidal embedding of integer timesteps: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class Conv3(nn.Module):
    """1D convolution (kernel 3, zero padding 1) along the chunk's time axis, written as one matrix multiply over each
    step and its two neighbours. Same maths as nn.Conv1d, much faster for such short sequences on CPU."""

    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(3 * cin, cout)

    def forward(self, x):  # (B, K, cin) -> (B, K, cout)
        p = torch.nn.functional.pad(x, (0, 0, 1, 1))  # zero-pad the time axis
        return self.lin(torch.cat([p[:, :-2], p[:, 1:-1], p[:, 2:]], dim=-1))


class ResBlock1D(nn.Module):
    """Temporal convolutions over the action chunk, modulated (FiLM) by a conditioning vector."""

    def __init__(self, ch, cond):
        super().__init__()
        self.c1, self.c2 = Conv3(ch, ch), Conv3(ch, ch)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.film = nn.Linear(cond, 2 * ch)
        self.act = nn.SiLU()

    def forward(self, x, cond):  # x: (B, K, ch), cond: (B, cond)
        h = self.c1(self.act(self.n1(x)))
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + scale[:, None, :]) + shift[:, None, :]
        return x + self.c2(self.act(self.n2(h)))


class Denoiser(nn.Module):
    """eps_theta(x_t, t, obs): predicts the noise that was added to an action chunk. x_t: (B, K, 3)."""

    def __init__(self, obs_dim, ch=64, cond=128, blocks=3):
        super().__init__()
        self.obs_enc = nn.Sequential(nn.Linear(obs_dim, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.t_enc = nn.Sequential(nn.Linear(64, cond), nn.SiLU(), nn.Linear(cond, cond))
        self.inp = Conv3(ACT_DIM, ch)
        self.blocks = nn.ModuleList([ResBlock1D(ch, 2 * cond) for _ in range(blocks)])
        self.out = Conv3(ch, ACT_DIM)

    def forward(self, x, t, obs):
        cond = torch.cat([self.obs_enc(obs), self.t_enc(timestep_embedding(t))], dim=-1)  # (B, 2*cond)
        h = self.inp(x)  # (B, K, ch)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(h)  # (B, K, 3)


class DiffusionPolicy(nn.Module):
    K = K_CHUNK

    def __init__(self, obs_dim, T=100, ddim_steps=10):
        super().__init__()
        self.net = Denoiser(obs_dim)
        self.sched = Schedule(T)
        self.ddim_steps = ddim_steps
        self.register_buffer("abar", torch.tensor(self.sched.abar, dtype=torch.float32))

    def loss(self, obs, chunk):
        """DDPM training: pick a random noise level, noise the true chunk, and regress the noise (epsilon-prediction)."""
        B = chunk.shape[0]
        t = torch.randint(0, self.sched.T, (B,), device=chunk.device)
        eps = torch.randn_like(chunk)
        ab = self.abar[t][:, None, None]
        x_t = ab.sqrt() * chunk + (1 - ab).sqrt() * eps
        return ((self.net(x_t, t, obs) - eps) ** 2).mean()

    @torch.no_grad()
    def sample(self, obs, x_T=None, steps=None, generator=None):
        """obs: (B, D) tensor -> (B, K, 3) tensor, DDIM with `steps` denoising steps."""
        B = obs.shape[0]
        if x_T is None:
            x_T = torch.randn(B, self.K, ACT_DIM, generator=generator)
        fn = lambda x, t, o: self.net(torch.from_numpy(x).float(), torch.from_numpy(t), o).numpy()
        out = ddim_sample(fn, self.sched, obs, (B, self.K, ACT_DIM), steps or self.ddim_steps, x_T.numpy())
        return torch.from_numpy(out).float()


def make_policy(kind, obs_dim):
    return {"mlp": MLPPolicy, "transformer": TransformerChunkPolicy, "diffusion": DiffusionPolicy}[kind](obs_dim)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class Agent:
    """Wraps a trained policy for rollouts: normalises the observation, predicts a chunk, executes the first
    n_exec actions, then re-plans. `use_ft=False` drops the 6 F/T columns."""

    def __init__(self, model, kind, use_ft, obs_mean, obs_std, seed=0, zero_ft=False):
        self.model, self.kind, self.use_ft = model.eval(), kind, use_ft
        self.mean, self.std = np.asarray(obs_mean, np.float32), np.asarray(obs_std, np.float32)
        self.n_exec = 1 if kind == "mlp" else N_EXEC
        self.gen = torch.Generator().manual_seed(seed)
        self.zero_ft = zero_ft

    def reset(self, env=None):
        self.queue = []

    @torch.no_grad()
    def act(self, obs, env=None):
        if not self.queue:
            o = np.array(obs, dtype=np.float32)
            if self.zero_ft:
                o[8:] = 0.0  # negative control: blank the F/T reading at test time
            if not self.use_ft:
                o = o[:8]
            x = torch.from_numpy((o - self.mean) / self.std)[None]
            chunk = (self.model.sample(x, generator=self.gen) if self.kind == "diffusion" else self.model(x))[0].numpy()
            self.queue = list(np.clip(chunk[: self.n_exec], -1, 1))
        return self.queue.pop(0)
