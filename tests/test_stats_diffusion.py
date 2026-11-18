import numpy as np
import torch

from clpl.data import make_chunks
from clpl.diffusion import Schedule, ddim_sample, ddim_timesteps, q_mean_std, q_sample
from clpl.policies import Agent, DiffusionPolicy
from clpl.stats import paired_bootstrap_diff, wilson


def test_wilson_hand_values():
    lo, hi = wilson(50, 100)  # textbook value: 0.404 .. 0.596
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = wilson(10, 10)  # all successes: lower bound 0.7225, upper bound 1
    assert abs(lo - 0.7225) < 1e-3 and hi == 1.0
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 1e-3


def test_paired_bootstrap_detects_a_real_difference_and_a_null():
    a, b = np.ones(100), np.zeros(100)
    d, (lo, hi) = paired_bootstrap_diff(a, b)
    assert d == 1.0 and lo == 1.0
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 400).astype(float)
    d, (lo, hi) = paired_bootstrap_diff(x, x.copy())
    assert d == 0.0 and lo == 0.0 and hi == 0.0


def test_schedule_matches_hand_formula():
    s = Schedule(100)
    # abar after step 1 by the cosine formula, computed independently here
    f = lambda t: np.cos((t / 100 + 0.008) / 1.008 * np.pi / 2) ** 2
    assert abs(s.abar[0] - f(1) / f(0)) < 1e-9
    assert np.all(np.diff(s.abar) < 0)  # strictly more noise each step
    assert s.abar[-1] < 1e-3  # last step is almost pure noise


def test_forward_noising_matches_closed_form_mean_and_variance():
    s = Schedule(100)
    rng = np.random.default_rng(0)
    x0 = np.array([0.7, -0.3, 0.5])[None, None, :] * np.ones((200_000, 2, 3))
    for t in (0, 30, 99):
        tt = np.full(200_000, t)
        xt = q_sample(s, x0, tt, rng.standard_normal(x0.shape))
        mean, std = q_mean_std(s, x0[0, 0], t)
        assert np.allclose(xt.mean(axis=0)[0], mean, atol=0.01)
        assert np.allclose(xt.std(axis=0)[0], std, atol=0.01)
        assert np.isclose(std**2, 1 - s.abar[t])  # variance is exactly 1 - abar


def test_ddim_with_oracle_noise_recovers_clean_chunk():
    """If the network were perfect (it returns the true noise), DDIM must return x0 for any number of steps."""
    s = Schedule(100)
    rng = np.random.default_rng(1)
    x0 = rng.uniform(-0.9, 0.9, size=(4, 8, 3))
    eps = rng.standard_normal(x0.shape)
    x_T = q_sample(s, x0, np.full(4, 99), eps)  # noise level of the first sampler step
    # the oracle reports the noise implied by the current x_t: eps = (x_t - sqrt(abar) x0) / sqrt(1 - abar)
    oracle = lambda x, t, obs: (x - np.sqrt(s.abar[t])[:, None, None] * x0) / np.sqrt(1 - s.abar[t])[:, None, None]
    for steps in (1, 5, 10, 50):
        out = ddim_sample(oracle, s, None, x0.shape, steps, x_T)
        assert np.allclose(out, x0, atol=1e-6), steps


def test_ddim_timesteps():
    assert list(ddim_timesteps(100, 5)) == [99, 74, 50, 25, 0]
    assert ddim_timesteps(100, 10)[0] == 99 and ddim_timesteps(100, 10)[-1] == 0


def test_diffusion_policy_loss_decreases_on_a_fixed_batch():
    torch.manual_seed(0)
    m = DiffusionPolicy(14)
    obs, chunk = torch.randn(64, 14), torch.rand(64, 8, 3) * 2 - 1
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    losses = []
    for _ in range(150):
        l = m.loss(obs, chunk)
        opt.zero_grad()
        l.backward()
        opt.step()
        losses.append(l.item())
    assert np.mean(losses[-20:]) < 0.8 * np.mean(losses[:20])
    out = m.sample(obs[:5], steps=5)
    assert out.shape == (5, 8, 3) and torch.isfinite(out).all() and out.abs().max() <= 1.0 + 1e-6


def test_make_chunks_pads_with_last_action_and_respects_episode_boundaries():
    act = np.array([[1.0], [2.0], [3.0], [10.0], [20.0]], dtype=np.float32)
    ep = np.array([0, 0, 0, 1, 1])
    obs = np.zeros((5, 2))
    c = make_chunks(obs, act, ep, 2)
    assert c[:, :, 0].tolist() == [[1, 2], [2, 3], [3, 3], [10, 20], [20, 20]]


def test_agent_replans_every_n_exec_steps():
    class Dummy(torch.nn.Module):
        calls = 0

        def forward(self, x):
            Dummy.calls += 1
            return torch.arange(8.0)[None, :, None].expand(1, 8, 3) / 10

    ag = Agent(Dummy(), "transformer", False, np.zeros(8), np.ones(8))
    ag.reset()
    acts = [ag.act(np.zeros(14))[0] for _ in range(8)]
    assert Dummy.calls == 2  # n_exec = 4 -> two model calls for 8 steps
    assert np.allclose(acts, [0, .1, .2, .3, 0, .1, .2, .3])
