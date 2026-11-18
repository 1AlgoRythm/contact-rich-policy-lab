"""Evaluate controllers and trained policies on fixed, shared eval seeds -> results/eval.json.

Episode seeds EVAL_SEED0 .. EVAL_SEED0+N-1 give the same hole position and the same unit-normal estimate error to every
variant, so comparisons are paired. The hole-noise level (sigma) only scales that estimate error.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.env import InsertionEnv  # noqa: E402
from clpl.expert import Expert, StraightToEstimate  # noqa: E402
from clpl.policies import Agent, make_policy  # noqa: E402

EVAL_SEED0 = 100_000
SIGMAS = [0.003, 0.005, 0.008]  # 3 mm = training noise; 5 and 8 mm are beyond the training range


def load_agent(name, zero_ft=False):
    ck = torch.load(f"models/{name}.pt", weights_only=False)
    obs_dim = len(ck["mean"])
    model = make_policy(ck["kind"], obs_dim)
    model.load_state_dict(ck["state"])
    seed = int(name.rsplit("_s", 1)[1])
    return Agent(model, ck["kind"], ck["use_ft"], ck["mean"], ck["std"], seed=seed, zero_ft=zero_ft)


def run_episodes(env, ctrl, n, sigma):
    ok, steps, force = [], [], []
    for i in range(n):
        obs = env.reset(EVAL_SEED0 + i, sigma)
        ctrl.reset(env)
        done = False
        while not done:
            obs, done, info = env.step(ctrl.act(obs, env))
        ok.append(int(info["success"]))
        steps.append(info["steps"])
        force.append(info["max_force"])
    return {"success": ok, "steps": steps, "peak_force": force}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    env = InsertionEnv()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    names = a.names or sorted(p.stem for p in Path("models").glob("*.pt"))
    jobs = [("straight_to_estimate", StraightToEstimate(), False), ("expert", Expert(seed=777), False)]
    for n in names:
        jobs.append((n, None, False))
        if "_ft_" in n:
            jobs.append((n + "+ftzero", None, True))  # negative control: F/T blanked at test time
    t0 = time.time()
    for key, ctrl, zero in jobs:
        for sigma in SIGMAS:
            if zero and sigma != SIGMAS[0]:
                continue
            ctrl_i = ctrl if ctrl is not None else load_agent(key.replace("+ftzero", ""), zero_ft=zero)
            if hasattr(ctrl_i, "rng"):
                ctrl_i.rng = np.random.default_rng(777)  # expert action-noise stream restarts per sigma
            r = run_episodes(env, ctrl_i, a.n, sigma)
            res.setdefault(key, {})[f"{sigma:.3f}"] = r
            print(f"{key:28s} sigma={sigma*1000:.0f}mm success={np.mean(r['success']):.3f} steps={np.mean(r['steps']):.1f} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))
    meta = {"n_episodes": a.n, "eval_seed0": EVAL_SEED0, "sigmas_m": SIGMAS, "wall_seconds": time.time() - t0,
            "loadavg_after": list(__import__("os").getloadavg())}
    res["_meta"] = meta
    out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
