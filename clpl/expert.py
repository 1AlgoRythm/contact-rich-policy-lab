"""Scripted controllers: the straight-to-estimate baseline and the privileged expert.

Both see the same things a policy sees (tip position, noisy hole estimate, F/T). The EXPERT is also handed the TRUE hole
position (privileged information, which no learned policy gets) and uses it only after the peg has touched the block.
"""
import numpy as np

from .env import ACT_SCALE

OBS_POS, OBS_VEL, OBS_REL, OBS_FT = slice(0, 3), slice(3, 6), slice(6, 8), slice(8, 14)


def _lateral(err_xy, gain=1.0):
    """Proportional move toward a point `err_xy` metres away, as an action in [-1, 1] (max 1 = 4 mm/step)."""
    return np.clip(gain * err_xy / ACT_SCALE, -1, 1)


class StraightToEstimate:
    """Baseline: slide over the noisy hole estimate, then push straight down. Never reacts to force."""

    def reset(self, env):
        pass

    def act(self, obs, env=None):
        rel = obs[OBS_REL]
        a = np.zeros(3)
        a[:2] = _lateral(rel, 0.5)
        a[2] = -1.0 if np.linalg.norm(rel) < 0.0007 or obs[2] < 0.01 else -0.2  # settle over the estimate first
        return a


class Expert:
    """Approach the noisy estimate like the baseline; after touchdown use the TRUE hole to slide into place.

    phase 0 approach: track the estimate, descend.
    phase 1 recover (after touching the block, vertical force > 1 N): lateral move toward the true hole while pressing
            gently (-0.3 -> about 2.4 N); this is the "search/compliance" behaviour.
    phase 2 insert (once within 0.6 mm of the true hole, or already below the surface): hold xy on the true hole, push down.
    Per-episode random speed factors and Gaussian action noise make demos differ from each other.
    """

    def __init__(self, action_noise=0.1, seed=0):
        self.rng = np.random.default_rng(seed)
        self.action_noise = action_noise

    def reset(self, env):
        self.contact = False
        self.speed = self.rng.uniform(0.6, 1.0)  # descent speed factor
        self.lat_gain = self.rng.uniform(0.3, 0.6)  # sliding speed factor

    def act(self, obs, env):
        tip, rel = obs[OBS_POS], obs[OBS_REL]
        true_err = env.hole - tip[:2]
        if abs(env.ft()[2]) > 1.0:
            self.contact = True
        a = np.zeros(3)
        if tip[2] < -0.002 or np.linalg.norm(true_err) < 0.0006:  # inserting
            a[:2] = _lateral(true_err, 0.5)
            a[2] = -self.speed
        elif self.contact:  # sliding on the surface / chamfer toward the true hole
            a[:2] = _lateral(true_err, self.lat_gain * 2)
            a[2] = -0.3
        else:  # approach over the estimate
            a[:2] = _lateral(rel, 0.5)
            a[2] = -self.speed if (np.linalg.norm(rel) < 0.0007 or tip[2] < 0.01) else -0.2
        return np.clip(a + self.rng.normal(size=3) * self.action_noise, -1, 1)


def rollout(env, controller, seed, hole_noise=None):
    obs = env.reset(seed, hole_noise)
    controller.reset(env)
    done = False
    while not done:
        obs, done, info = env.step(controller.act(obs, env))
    return info
