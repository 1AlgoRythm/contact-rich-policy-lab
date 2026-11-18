"""MuJoCo peg-in-hole with a wrist force/torque sensor.

Physics in plain words (full description in README):
  * A square peg (20 mm wide, 50 mm long) is carried by a "hand" that can slide along x, y, z.
    Each axis has a stiff spring-damper position actuator (a PD controller), like a position-controlled arm.
  * A block has a square hole with a 45 degree chamfer (a sloped funnel) around its mouth.
  * The peg is a separate body welded to the hand (no joint). A MuJoCo force sensor and torque sensor sit on that
    weld, so they read the load passing from the peg to the hand: contact force plus the peg's own inertia (6 numbers).
    This is deliberate: a sensor on the hand itself would also read the actuator's drive force.
  * Gravity is off (the peg's weight is assumed compensated), so the F/T reading is contact only.
Control: 20 Hz. One action = a displacement command: the position setpoint moves by action * 4 mm per step. The setpoint
may lead the measured position by at most 5 mm (so the arm can push with at most 2000 N/m * 5 mm = 10 N per axis).
Because the setpoint persists, a pressed peg feels a spring force, like a stiff position-controlled arm.
"""
import mujoco
import numpy as np

CONTROL_HZ = 20
TIMESTEP = 0.002
SUBSTEPS = int(round(1 / CONTROL_HZ / TIMESTEP))  # 25 physics steps per control step
ACT_SCALE = 0.004  # metres of setpoint motion per unit action
MAX_LEAD = 0.005  # setpoint may lead the measured position by at most this (anti wind-up)
PEG_HALF = 0.01  # peg half-width (m)
PEG_HALF_LEN = 0.025  # peg half-length (m)
HOLE_DEPTH = 0.03
SUCCESS_DEPTH = 0.015  # peg tip must be this far below the block top
HOLE_RANGE = 0.008  # true hole centre ~ U(-8 mm, 8 mm) per axis around the origin
START_Z = 0.03  # peg tip starts this high above the block top
OBS_DIM = 8  # tip pos (3), tip vel (3), noisy hole estimate minus tip xy (2)
FT_DIM = 6


def build_xml(clearance=0.001, chamfer=0.004, friction=0.3):
    hw = PEG_HALF + clearance  # hole half-width
    W, D, c, t = 0.05, HOLE_DEPTH, chamfer, 0.01
    s2 = np.sqrt(2.0)
    g = []

    def box(name, pos, half, euler="0 0 0"):
        g.append(f'<geom name="{name}" type="box" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
                 f'size="{half[0]:.6f} {half[1]:.6f} {half[2]:.6f}" euler="{euler}" friction="{friction} 0.005 0.0001"/>')

    # Solid blocks around the hole. Each side wall is split into an outer part (full height) and an inner strip
    # (top at -c) so the chamfer wedge is empty; a rotated slab then forms the sloped face.
    ow, iw = W - hw - c, c
    for sx, nm in ((-1, "xm"), (1, "xp")):
        box(f"{nm}_out", (sx * (hw + c + ow / 2), 0, -D / 2), (ow / 2, W, D / 2))
        box(f"{nm}_in", (sx * (hw + iw / 2), 0, -(c + D) / 2), (iw / 2, W, (D - c) / 2))
        L = c * s2 / 2  # half slope length
        mid = (sx * (hw + c / 2), -c / 2)
        ctr = (mid[0] + sx * t / s2, mid[1] - t / s2)
        box(f"{nm}_slope", (ctr[0], 0, ctr[1]), (L, W, t), euler=f"0 {-45 * sx} 0")
    # +-y sides span only the x extent of the hole (+ chamfer) to avoid redundant geoms
    xs = hw + c
    for sy, nm in ((-1, "ym"), (1, "yp")):
        box(f"{nm}_out", (0, sy * (hw + c + ow / 2), -D / 2), (xs, ow / 2, D / 2))
        box(f"{nm}_in", (0, sy * (hw + iw / 2), -(c + D) / 2), (xs, iw / 2, (D - c) / 2))
        L = c * s2 / 2
        mid = (sy * (hw + c / 2), -c / 2)
        ctr = (mid[0] + sy * t / s2, mid[1] - t / s2)
        box(f"{nm}_slope", (0, ctr[0], ctr[1]), (xs, L, t), euler=f"{45 * sy} 0 0")
    box("floor", (0, 0, -D - 0.005), (W, W, 0.005))
    geoms = "\n      ".join(g)
    return f"""
<mujoco model="peg_in_hole">
  <option timestep="{TIMESTEP}" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="block" pos="0 0 0">
      {geoms}
    </body>
    <body name="hand" pos="0 0 {START_Z + PEG_HALF_LEN}">
      <joint name="jx" type="slide" axis="1 0 0" damping="0" armature="0.01"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="0" armature="0.01"/>
      <joint name="jz" type="slide" axis="0 0 1" damping="0" armature="0.01"/>
      <inertial pos="0 0 0" mass="0.5" diaginertia="0.001 0.001 0.001"/>
      <body name="peg">
        <site name="ft" pos="0 0 0" size="0.002"/>
        <geom name="peg" type="box" size="{PEG_HALF} {PEG_HALF} {PEG_HALF_LEN}" mass="0.1" friction="{friction} 0.005 0.0001"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="jx" kp="2000" kv="80"/>
    <position name="ay" joint="jy" kp="2000" kv="80"/>
    <position name="az" joint="jz" kp="2000" kv="80"/>
  </actuator>
  <sensor>
    <force name="f" site="ft"/>
    <torque name="tq" site="ft"/>
  </sensor>
</mujoco>"""


class InsertionEnv:
    def __init__(self, hole_noise=0.003, max_steps=100, clearance=0.001, chamfer=0.004, friction=0.3,
                 force_noise=0.05, torque_noise=0.002):
        self.m = mujoco.MjModel.from_xml_string(build_xml(clearance, chamfer, friction))
        self.d = mujoco.MjData(self.m)
        self.block_id = self.m.body("block").id
        self.hole_noise, self.max_steps = hole_noise, max_steps
        self.force_noise, self.torque_noise = force_noise, torque_noise
        self.fsl = slice(self.m.sensor_adr[0], self.m.sensor_adr[0] + 3)
        self.tsl = slice(self.m.sensor_adr[1], self.m.sensor_adr[1] + 3)
        self.hole = np.zeros(2)
        self.est = np.zeros(2)

    # -- helpers -----------------------------------------------------------------------------------------
    def _tip_pos(self):
        # hand body origin = peg centre; qpos is the slide offset from the body's home (0, 0, START_Z + HALF_LEN)
        return np.array([self.d.qpos[0], self.d.qpos[1], self.d.qpos[2] + START_Z])

    def _tip_vel(self):
        return self.d.qvel[:3].copy()

    def ft(self):
        """Noise-free 6-number wrist reading (N, N, N, Nm, Nm, Nm)."""
        return np.concatenate([self.d.sensordata[self.fsl], self.d.sensordata[self.tsl]])

    def _obs(self):
        tip, vel = self._tip_pos(), self._tip_vel()
        ft = self.ft() + self.rng.normal(size=6) * np.r_[[self.force_noise] * 3, [self.torque_noise] * 3]
        self.last_ft = ft
        return np.concatenate([tip, vel, self.est - tip[:2], ft]).astype(np.float32)

    # -- gym-like API ------------------------------------------------------------------------------------
    def reset(self, seed, hole_noise=None):
        """All randomness derives from `seed`, so the same seed gives the same hole across policies and noise levels."""
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 10_000_019)  # sensor noise stream
        sigma = self.hole_noise if hole_noise is None else hole_noise
        self.hole = rng.uniform(-HOLE_RANGE, HOLE_RANGE, size=2)
        self.est = self.hole + sigma * rng.normal(size=2)  # unit-normal draw is fixed by the seed; sigma only scales it
        start_xy = rng.uniform(-0.002, 0.002, size=2)
        mujoco.mj_resetData(self.m, self.d)
        self.m.body_pos[self.block_id] = [self.hole[0], self.hole[1], 0.0]
        self.d.qpos[:2] = start_xy
        self.target = np.array([start_xy[0], start_xy[1], 0.0])
        self.d.ctrl[:] = self.target
        mujoco.mj_forward(self.m, self.d)
        self.t, self.max_force, self.success = 0, 0.0, False
        return self._obs()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        q = self.d.qpos[:3]
        self.target = np.clip(self.target + a * ACT_SCALE, q - MAX_LEAD, q + MAX_LEAD)  # setpoint integrates the action
        self.d.ctrl[:] = self.target
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        self.max_force = max(self.max_force, float(np.linalg.norm(self.ft()[:3])))
        tip = self._tip_pos()
        self.success = bool(tip[2] < -SUCCESS_DEPTH)
        done = self.success or self.t >= self.max_steps or not np.all(np.isfinite(self.d.qpos))
        return self._obs(), done, {"success": self.success, "max_force": self.max_force, "steps": self.t}
