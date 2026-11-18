"""Success of the diffusion policy vs number of DDIM denoising steps (in PyTorch, sigma = 3 mm, shared eval seeds)
-> results/ddim_steps.json. Pairs with the latency-vs-steps table in results/onnx.json."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.env import InsertionEnv  # noqa: E402
from clpl.stats import wilson  # noqa: E402
from scripts.evaluate import EVAL_SEED0, load_agent, run_episodes  # noqa: E402

N, STEPS = 100, [1, 2, 5, 10, 20]


def main():
    torch.set_num_threads(2)
    env, res = InsertionEnv(), {}
    for tag in ("ft", "noft"):
        for S in STEPS:
            runs = []
            for seed in (0, 1, 2):
                ag = load_agent(f"diffusion_{tag}_s{seed}")
                ag.model.ddim_steps = S
                runs.append(run_episodes(env, ag, N, 0.003)["success"])
            k, n = int(np.sum(runs)), np.size(runs)
            lo, hi = wilson(k, n)
            res[f"diffusion_{tag}_ddim{S}"] = {"success": k / n, "wilson95": [lo, hi], "per_seed": np.mean(runs, 1).tolist()}
            print(tag, S, k / n, flush=True)
    res["_meta"] = {"n_episodes_per_seed": N, "eval_seed0": EVAL_SEED0, "sigma_m": 0.003}
    Path("results/ddim_steps.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
