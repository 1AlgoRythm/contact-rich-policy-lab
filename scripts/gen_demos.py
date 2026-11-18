"""Generate scripted expert demonstrations (successes only) -> data/demos.npz and results/demos.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.env import InsertionEnv  # noqa: E402
from clpl.expert import Expert  # noqa: E402
from clpl.stats import wilson  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of successful demos to keep")
    ap.add_argument("--seed0", type=int, default=0, help="first episode seed (eval seeds start at 100000)")
    ap.add_argument("--out", default="data/demos.npz")
    a = ap.parse_args()
    t0 = time.time()
    env, expert = InsertionEnv(), Expert(seed=12345)
    obs_l, act_l, ep_l, lens = [], [], [], []
    attempts = successes = 0
    seed = a.seed0
    while successes < a.n:
        o = env.reset(seed)
        expert.reset(env)
        O, A, done = [], [], False
        while not done:
            act = expert.act(o, env)
            O.append(o)
            A.append(act)
            o, done, info = env.step(act)
        attempts += 1
        if info["success"]:
            obs_l.append(np.array(O))
            act_l.append(np.array(A, dtype=np.float32))
            ep_l.append(np.full(len(O), successes))
            lens.append(len(O))
            successes += 1
        seed += 1
    Path(a.out).parent.mkdir(exist_ok=True)
    np.savez_compressed(a.out, obs=np.concatenate(obs_l), act=np.concatenate(act_l), ep=np.concatenate(ep_l))
    lo, hi = wilson(successes, attempts)
    res = {"demos_kept": successes, "attempts": attempts, "expert_success_rate": successes / attempts,
           "expert_success_wilson95": [lo, hi], "samples": int(sum(lens)), "mean_len": float(np.mean(lens)),
           "seed_range": [a.seed0, seed - 1], "hole_noise_m": env.hole_noise, "wall_seconds": time.time() - t0}
    Path("results").mkdir(exist_ok=True)
    Path("results/demos.json").write_text(json.dumps(res, indent=2))
    print(res)


if __name__ == "__main__":
    main()
