import numpy as np

from clpl.env import ACT_SCALE, HOLE_RANGE, InsertionEnv, MAX_LEAD
from clpl.expert import Expert, StraightToEstimate, rollout


def test_same_seed_is_deterministic():
    runs = []
    for _ in range(2):
        env = InsertionEnv()
        o = env.reset(5)
        traj = [o]
        for k in range(30):
            o, d, _ = env.step([0.3, -0.2, -0.5])
            traj.append(o)
        runs.append(np.array(traj))
    assert np.array_equal(runs[0], runs[1])


def test_sigma_only_scales_the_estimate_error():
    env = InsertionEnv()
    env.reset(7, 0.003)
    hole, e3 = env.hole.copy(), env.est - env.hole
    env.reset(7, 0.006)
    assert np.allclose(env.hole, hole) and np.allclose(env.est - env.hole, 2 * e3)
    env.reset(7, 0.0)
    assert np.allclose(env.est, env.hole)


def test_hole_inside_range_and_peg_starts_30mm_above():
    env = InsertionEnv()
    for s in range(20):
        o = env.reset(s)
        assert np.all(np.abs(env.hole) <= HOLE_RANGE)
        assert abs(o[2] - 0.03) < 1e-6 and np.allclose(o[3:6], 0)


def test_ft_reads_the_pressing_force_hand_computed():
    """Press straight down on the flat top, far from the hole. At rest the setpoint leads the peg by MAX_LEAD, so the
    spring force is kp * MAX_LEAD = 2000 N/m * 0.005 m = 10 N, and the block pushes back with the same 10 N."""
    env = InsertionEnv()
    env.reset(0)
    env.hole = env.hole  # unchanged; start xy is within 2 mm of the origin, the hole is >= 0 mm away
    env.m.body_pos[env.block_id] = [0.04, 0.04, 0.0]  # move the hole far away so the peg lands on flat top
    import mujoco
    mujoco.mj_forward(env.m, env.d)
    for _ in range(40):
        o, d, info = env.step([0, 0, -1])
    ft = env.ft()
    assert abs(abs(ft[2]) - 10.0) < 0.5
    assert abs(ft[0]) < 0.1 and abs(ft[1]) < 0.1  # no sideways force on a flat surface
    assert abs(o[2]) < 5e-4  # tip rests on the surface (z = 0), penetration under 0.5 mm
    assert not info["success"]


def test_aligned_peg_goes_in_and_misaligned_peg_stays_out():
    env = InsertionEnv(hole_noise=0.0)

    def go(offset):
        o = env.reset(3, 0.0)
        for _ in range(60):
            err = env.hole + offset - o[:2]
            a = np.r_[np.clip(err / ACT_SCALE * 0.5, -1, 1), -1.0 if np.linalg.norm(err) < 7e-4 or o[2] < 0.01 else -0.2]
            o, d, info = env.step(a)
            if d:
                break
        return info["success"]

    assert go(np.zeros(2)) is True
    assert go(np.array([0.008, 0.0])) is False  # 8 mm off is outside peg + clearance + chamfer


def test_sim_is_stable_under_violent_actions():
    env = InsertionEnv()
    rng = np.random.default_rng(0)
    for s in range(10):
        o = env.reset(s)
        for _ in range(100):
            o, d, _ = env.step(rng.choice([-1.0, 1.0], size=3))
            assert np.all(np.isfinite(o))
            assert np.abs(env.d.qvel[:3]).max() < 3.0  # m/s
            assert np.abs(env.d.qpos[:3]).max() < 0.3
            if d:
                break
    # setpoint can never lead the arm by more than MAX_LEAD
    assert np.abs(env.target - env.d.qpos[:3]).max() <= MAX_LEAD + 1e-9


def test_expert_beats_baseline_and_privilege_matters():
    env = InsertionEnv()
    seeds = range(200_000, 200_060)
    base = np.mean([rollout(env, StraightToEstimate(), s)["success"] for s in seeds])
    exp = np.mean([rollout(env, Expert(seed=1), s)["success"] for s in seeds])
    assert exp >= 0.95 and base < 0.6 and exp > base + 0.4

    class Blind(Expert):  # deliberate break: hand the "expert" the noisy estimate instead of the true hole
        def act(self, obs, env_):
            class Fake:
                hole = env_.est
                ft = env_.ft

            return super().act(obs, Fake())

    blind = np.mean([rollout(env, Blind(seed=1), s)["success"] for s in seeds])
    assert blind < exp - 0.4  # without privileged information the expert collapses


def test_ft_is_a_wrist_sensor_not_an_actuator_readout():
    """In free flight the reading must be near zero (only the 0.1 kg peg's inertia, about m*a < 0.5 N), even though the
    actuators push with about 1 N while accelerating the 0.6 kg hand+peg."""
    env = InsertionEnv(force_noise=0.0, torque_noise=0.0)
    env.reset(0)
    peak_actuator = peak_ft = 0.0
    for _ in range(6):
        env.step([0, 0, -1])
        peak_ft = max(peak_ft, float(np.linalg.norm(env.ft()[:3])))
        peak_actuator = max(peak_actuator, float(np.abs(env.d.actuator_force).max()))
    assert peak_actuator > 1.0 and peak_ft < 0.5
