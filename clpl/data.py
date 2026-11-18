"""Load demos and turn them into (observation, action-chunk) training pairs."""
import numpy as np

N_VAL_EPISODES = 50  # last 50 of the 500 demo episodes are held out (only to report validation loss)


def load_demos(path="data/demos.npz"):
    z = np.load(path)
    return z["obs"], z["act"], z["ep"]


def make_chunks(obs, act, ep, K):
    """For every step t, the chunk is actions t..t+K-1 of the same episode; past the end, repeat the last action."""
    chunks = np.zeros((len(obs), K, act.shape[1]), dtype=np.float32)
    for e in np.unique(ep):
        idx = np.where(ep == e)[0]
        a = act[idx]
        for k in range(K):
            j = np.minimum(np.arange(len(idx)) + k, len(idx) - 1)
            chunks[idx, k] = a[j]
    return chunks


def split(ep):
    val = ep >= ep.max() + 1 - N_VAL_EPISODES
    return ~val, val
