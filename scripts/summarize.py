"""results/eval.json (per-episode outcomes) -> results/summary.json (rates, intervals, paired comparisons).

Pooling: for a trained variant, the 3 training seeds are pooled (3 x N episodes) for the Wilson interval, and the
per-seed rates are also reported as mean +/- std. Because every seed sees the SAME N eval episodes, the pooled episodes
are not independent draws: the pooled Wilson interval is a little optimistic; the std over seeds is the more honest
spread of training randomness. Paired comparisons average each episode's outcome over seeds, then bootstrap episodes.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.stats import mean_std, paired_bootstrap_diff, wilson  # noqa: E402

KINDS = ["mlp", "transformer", "diffusion"]
SIGMAS = ["0.003", "0.005", "0.008"]


def block(runs, sigma):
    """Aggregate a list of per-seed result dicts at one sigma."""
    succ = np.array([r[sigma]["success"] for r in runs])  # (seeds, N)
    steps = np.array([r[sigma]["steps"] for r in runs], float)
    force = np.array([r[sigma]["peak_force"] for r in runs], float)
    k, n = int(succ.sum()), succ.size
    lo, hi = wilson(k, n)
    per_seed = succ.mean(axis=1)
    m, s = mean_std(per_seed)
    ok = succ.astype(bool)
    return {"n_seeds": len(runs), "n_episodes_per_seed": succ.shape[1], "pooled_successes": k, "pooled_n": n,
            "success": k / n, "wilson95": [lo, hi], "per_seed_success": per_seed.tolist(), "seed_mean": m, "seed_std": s,
            "steps_to_insert_mean": float(steps[ok].mean()) if ok.any() else None,
            "peak_force_mean_N": float(force.mean()), "peak_force_p95_N": float(np.percentile(force, 95)),
            "per_episode_success_mean_over_seeds": succ.mean(axis=0).tolist()}


def main():
    ev = json.loads(Path("results/eval.json").read_text())
    out = {"controllers": {}, "variants": {}, "ftzero": {}, "paired_sigma0.003": {}, "meta": ev.get("_meta")}
    for name in ("straight_to_estimate", "expert"):
        out["controllers"][name] = {s: block([ev[name]], s) for s in SIGMAS}
    for kind in KINDS:
        for ft in (True, False):
            runs = [ev[k] for k in sorted(ev) if k.startswith(f"{kind}_{'ft' if ft else 'noft'}_s") and "+" not in k]
            out["variants"][f"{kind}_{'ft' if ft else 'noft'}"] = {s: block(runs, s) for s in SIGMAS}
        zr = [ev[k] for k in sorted(ev) if k.startswith(f"{kind}_ft_s") and k.endswith("+ftzero")]
        out["ftzero"][f"{kind}_ft"] = {"0.003": block(zr, "0.003")}
    pe = lambda key: np.array(out["variants"][key]["0.003"]["per_episode_success_mean_over_seeds"])
    base = np.array(out["controllers"]["straight_to_estimate"]["0.003"]["per_episode_success_mean_over_seeds"])
    pairs = {}
    for kind in KINDS:
        pairs[f"{kind}: ft minus noft"] = (pe(f"{kind}_ft"), pe(f"{kind}_noft"))
        pairs[f"{kind}_ft minus straight_to_estimate"] = (pe(f"{kind}_ft"), base)
        pairs[f"{kind}_noft minus straight_to_estimate"] = (pe(f"{kind}_noft"), base)
    for a, b in (("diffusion", "mlp"), ("diffusion", "transformer"), ("transformer", "mlp")):
        for tag in ("ft", "noft"):
            pairs[f"{a}_{tag} minus {b}_{tag}"] = (pe(f"{a}_{tag}"), pe(f"{b}_{tag}"))
    for k, (a, b) in pairs.items():
        d, (lo, hi) = paired_bootstrap_diff(a, b)
        out["paired_sigma0.003"][k] = {"diff": d, "ci95": [lo, hi]}
    best = max(out["variants"], key=lambda v: out["variants"][v]["0.003"]["success"])
    out["best_variant_at_sigma0.003"] = best
    Path("results/summary.json").write_text(json.dumps(out, indent=1))
    for v, d in out["variants"].items():
        print(f"{v:18s}", " ".join(f"{s}: {d[s]['success']:.3f} [{d[s]['wilson95'][0]:.3f},{d[s]['wilson95'][1]:.3f}]" for s in SIGMAS))
    print("best:", best)


if __name__ == "__main__":
    main()
