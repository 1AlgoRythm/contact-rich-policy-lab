"""Sanity checks of the simulator -> results/env_check.json: stability, F/T sensor vs hand calculation, sim speed,
and how the straight-to-estimate baseline's success depends on clearance and chamfer (task neither trivial nor impossible)."""
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clpl.env import InsertionEnv, MAX_LEAD  # noqa: E402
from clpl.expert import StraightToEstimate, rollout  # noqa: E402
from clpl.stats import wilson  # noqa: E402


def main():
    res = {"mujoco": mujoco.__version__}
    env = InsertionEnv()
    # 1. stability under saturated random actions (+-1 on every axis, every step)
    rng = np.random.default_rng(0)
    vmax = zmin = 0.0
    nonfinite = steps = 0
    t0 = time.time()
    for s in range(50):
        env.reset(900_000 + s)
        for _ in range(100):
            o, d, _ = env.step(rng.choice([-1.0, 1.0], size=3))
            steps += 1
            nonfinite += int(not np.all(np.isfinite(o)))
            vmax = max(vmax, float(np.abs(env.d.qvel[:3]).max()))
            zmin = min(zmin, float(o[2]))
            if d:
                break
    res["stability"] = {"episodes": 50, "control_steps": steps, "nonfinite_obs": nonfinite, "max_abs_velocity_m_s": vmax,
                        "lowest_tip_z_m": zmin, "control_steps_per_second": steps / (time.time() - t0)}
    # 2. F/T against a hand calculation: pressing on flat top gives kp * MAX_LEAD = 2000 * 0.005 = 10 N
    env.reset(0)
    env.m.body_pos[env.block_id] = [0.04, 0.04, 0.0]
    mujoco.mj_forward(env.m, env.d)
    for _ in range(40):
        env.step([0, 0, -1])
    res["ft_press_test"] = {"expected_fz_N": 2000 * MAX_LEAD, "measured_fz_N": float(env.ft()[2]),
                            "measured_fx_N": float(env.ft()[0]), "measured_fy_N": float(env.ft()[1])}
    # 3. baseline success vs clearance and chamfer (sigma = 3 mm, 100 eval-seed episodes)
    grid = {}
    for cl in (0.0005, 0.001, 0.002):
        for ch in (0.002, 0.004, 0.006):
            e = InsertionEnv(clearance=cl, chamfer=ch)
            k = sum(rollout(e, StraightToEstimate(), 100_000 + i)["success"] for i in range(100))
            lo, hi = wilson(k, 100)
            grid[f"clearance_{cl*1000:g}mm_chamfer_{ch*1000:g}mm"] = {"success": k / 100, "wilson95": [lo, hi]}
    res["baseline_vs_geometry_sigma3mm"] = grid
    res["chosen"] = {"clearance_mm": 1.0, "chamfer_mm": 4.0, "note": "kept from a first probe of 4 geometries (baseline 17-29% in all); picked before any policy was trained"}
    Path("results/env_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
