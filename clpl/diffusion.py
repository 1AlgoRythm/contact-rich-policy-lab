"""DDPM noise schedule, forward noising, and a DDIM sampler, written from the equations (no diffusion library).

Notation, in words. x0 is a clean action chunk. At step t we mix it with Gaussian noise:
    x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)
so x_t has mean sqrt(abar_t) * x0 and variance (1 - abar_t). abar_t ("alpha bar") is the running product of (1 - beta_s).
The network is trained to predict eps from (x_t, t, observation) (epsilon-prediction, DDPM training).
"""
import numpy as np


def cosine_abar(T, s=0.008):
    """abar_0..abar_T from the cosine schedule of Nichol & Dhariwal 2021 (formula from memory, not checked)."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]


class Schedule:
    def __init__(self, T=100):
        self.T = T
        abar_full = cosine_abar(T)
        self.betas = np.clip(1 - abar_full[1:] / abar_full[:-1], 0, 0.999)
        self.abar = np.cumprod(1 - self.betas)  # abar[t] for t = 0..T-1 (index t means noise level of step t+1)


def q_mean_std(sched, x0, t):
    """Closed-form mean and standard deviation of x_t given x0 (numpy)."""
    ab = sched.abar[t]
    return np.sqrt(ab) * x0, np.sqrt(1 - ab)


def q_sample(sched, x0, t, eps):
    """Forward noising, numpy. x0, eps: (B, K, A); t: (B,) ints."""
    ab = sched.abar[t][:, None, None]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps


def ddim_timesteps(T, steps):
    """`steps` timesteps from T-1 down to 0, evenly spaced."""
    return np.unique(np.round(np.linspace(0, T - 1, steps)).astype(int))[::-1]


def ddim_sample(eps_fn, sched, obs, shape, steps, x_T, clip=1.0):
    """Deterministic DDIM sampling (eta = 0). eps_fn(x_t (B,K,A), t (B,) int, obs) -> eps_hat, numpy in and out.
    Each step: predict eps, recover x0_hat = (x_t - sqrt(1-abar) eps) / sqrt(abar), clip it, then re-noise it to the
    previous (lower) noise level with the SAME eps: x_prev = sqrt(abar_prev) x0_hat + sqrt(1 - abar_prev) eps."""
    x = x_T
    ts = ddim_timesteps(sched.T, steps)
    for i, t in enumerate(ts):
        tb = np.full((x.shape[0],), t, dtype=np.int64)
        eps = eps_fn(x, tb, obs)
        ab = sched.abar[t]
        x0 = (x - np.sqrt(1 - ab) * eps) / np.sqrt(ab)
        if clip is not None:
            x0 = np.clip(x0, -clip, clip)
        ab_prev = sched.abar[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = np.sqrt(ab_prev) * x0 + np.sqrt(1 - ab_prev) * eps
    return x
