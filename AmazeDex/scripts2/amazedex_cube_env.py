
import os
from dataclasses import dataclass

import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_env import MujocoEnv

ACTUATORS = ["motor_finger1_1", "motor_finger1_2", "motor_finger2_1", "motor_finger2_2",
             "motor_finger3_1", "motor_finger3_2", "motor_finger4_1", "motor_finger4_2"]
JOINTS = ["finger1_motor1", "finger1_motor2", "finger2_motor1", "finger2_motor2",
          "finger3_motor1", "finger3_motor2", "finger4_motor1", "finger4_motor2"]
TIP_SITES = ["tip1", "tip2", "tip3", "tip4"]
N_JOINTS = 8

MODEL_PATH = os.environ.get(
    "AMAZEDEX_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.xml"),
)

FACE_NORMALS = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
                         dtype=np.float32)
FACE_NAMES = ["+Z", "-Z", "+Y", "-Y", "+X", "-X"]
DICE_NUMBER = [5, 0, 4, 2, 1, 3]  # DICE_NUMBER[face_index] -> printed number

CUBE_LOCAL_CENTER = np.array([0, 0, 0], dtype=np.float32)  # calibrated cube-body -> geometric-center offset

# Observation scaling (keeps every obs component roughly in [-1, 1]).
JVEL_SCALE = 5.0

# Grasp closing.
HAND_CLOSE_FRAC = 0.63
HAND_CLOSE_JITTER = 0.005
CLOSE_PROBE_FRAC = 0.3
GRASP_OPEN_FRAC = 0.05
SETTLE_STEPS = 30
MAX_CTRL_RATE_FRAC = 0.16  # hard servo-speed limit, independent of the policy


@dataclass(frozen=True)
class Cfg:
    max_steps: int = 500

    # -- drop detection (persistence required so a single contact spike can't
    #    trigger it -- guardrail against noisy false terminations) --
    xy_drop_m: float = 0.18
    z_drop_m: float = 0.18
    drop_persist_steps: int = 6

    # -- reset randomization --
    pos_jitter_m: float = 0.0001
    yaw_jitter_rad: float = float(np.radians(1.0))

    # -- low-level control shaping (NOT reward -- this just keeps actions
    #    from teleporting the servos) --
    grasp_band_frac: float = 0.80
    action_lpf: float = 0.6
    max_ctrl_rate_frac: float = MAX_CTRL_RATE_FRAC

    # === the only 3 reward terms ===
    k_rotate: float = 3.9              # alpha_1
    rotate_clip_rad: float = 0.30      # clip per-step progress -- guardrail, see _reward()
    success_bonus: float = 16.0        # alpha_2
    drop_penalty: float = 20.0         # alpha_3 (applied as a flat, one-time negative)

    # -- success detection hysteresis (guardrail against reward hacking by
    #    oscillating across the threshold to farm the bonus repeatedly) --
    success_theta_rad: float = 0.50
    success_hold_steps: int = 2
    success_max_angvel: float = 4.0
    success_rearm_theta_rad: float = 0.6
    min_steps_between_success: int = 2


CFG = Cfg()


# ---------------------------------------------------------------------------
# Minimal quaternion helpers.
# ---------------------------------------------------------------------------
def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _q_from_vecs(v_from, v_to):
    v_from, v_to = v_from / np.linalg.norm(v_from), v_to / np.linalg.norm(v_to)
    d = float(np.dot(v_from, v_to))
    if d > 1 - 1e-8:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-8:
        axis = np.cross(v_from, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(v_from, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])
    axis = np.cross(v_from, v_to)
    q = np.array([1 + d, *axis])
    return q / np.linalg.norm(q)


def _require(model, kind, names):
    """Fail loudly and specifically instead of a cryptic mujoco KeyError,
    if robot.xml / scene.xml drift out of sync with this file."""
    getter = {"joint": model.joint, "actuator": model.actuator,
              "site": model.site, "body": model.body}[kind]
    for n in names:
        try:
            getter(n)
        except KeyError:
            raise KeyError(f"{kind} '{n}' not found in the MJCF model. "
                            f"robot.xml / scene.xml no longer matches amazedex_cube_env.py.")


class AmazeDexCubeEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, model_path=MODEL_PATH, render_mode=None, randomize=False, cfg=CFG, **kw):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__(model_path, frame_skip=10, render_mode=render_mode)
        self.cfg = cfg
        self.randomize = randomize

        _require(self.model, "actuator", ACTUATORS)
        _require(self.model, "joint", JOINTS)
        _require(self.model, "site", TIP_SITES)
        _require(self.model, "body", ["cube"])

        self.actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvelids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        cj = self.model.body("cube").jntadr[0]
        self.cube_qpos = self.model.jnt_qposadr[cj]
        self.cube_dof = self.model.jnt_dofadr[cj]

        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        tip_bodies = [self.model.site_bodyid[s] for s in self.tip_sites]
        self.tip_geoms = [{g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b}
                           for b in tip_bodies]
        self.cube_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.cubeid}
        self.base_friction = self.model.geom_friction.copy()

        # Palm-center calibration: whatever position the cube is authored at
        # in scene.xml (geometric center of the palm, per the MJCF) is taken
        # as the nominal spawn point every reset.
        self.nominal_cube_center = (self.init_qpos[self.cube_qpos:self.cube_qpos + 3].copy()
                                     + CUBE_LOCAL_CENTER)

        lims = self.model.actuator_ctrlrange[self.actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
        # 8 joint pos + 8 joint vel + 8 last action + 3 cube "up" (local frame)
        # + 6 target one-hot. The last 9 are the minimum goal signal a
        # goal-conditioned policy needs -- everything else stays out of obs.
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(33,), dtype=np.float32)

        self.last_action = np.zeros(N_JOINTS, np.float32)
        self.filtered_ctrl = np.zeros(N_JOINTS, np.float32)
        self.grasp_frac = np.zeros(N_JOINTS, np.float32)
        self.step_count = 0
        self.target_face = 0
        self.start_face = 0
        self.start_pos = np.zeros(3)
        self.prev_theta = np.pi
        self._armed = True
        self._hold = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._last_success_step = -10_000
        self.close_sign = self._detect_close_sign()

    # -- geometry -----------------------------------------------------------
    def _cube_center_world(self):
        R = self.data.xmat[self.cubeid].reshape(3, 3)
        return self.data.xpos[self.cubeid] + R @ CUBE_LOCAL_CENTER

    def _detect_close_sign(self):
        """Probe both actuation directions once at load time to learn which
        sign of ctrl actually closes the fingers on *this* MJCF, instead of
        hardcoding it."""
        qpos_save = self.data.qpos.copy()
        dists = {}
        for sign in (1.0, -1.0):
            self.data.qpos[self.qposids] = self.ctrl_mid + sign * CLOSE_PROBE_FRAC * self.ctrl_half
            mujoco.mj_forward(self.model, self.data)
            cpos = self._cube_center_world()
            dists[sign] = float(np.mean([np.linalg.norm(self.data.site_xpos[s] - cpos)
                                          for s in self.tip_sites]))
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        return 1.0 if dists[1.0] < dists[-1.0] else -1.0

    @property
    def targetface(self):
        return self.target_face

    @targetface.setter
    def targetface(self, value):
        self.target_face = int(value)
        self.prev_theta = self._theta()
        self._armed = True
        self._hold = 0

    # -- reset ----------------------------------------------------------------
    def reset_model(self):
        c = self.cfg
        qpos_open = np.clip(self.ctrl_mid + self.close_sign * GRASP_OPEN_FRAC * self.ctrl_half,
                             self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qposids] = qpos_open
        self.data.qvel[self.qvelids] = 0.0
        self.data.ctrl[self.actids] = qpos_open

        base_quat = self.init_qpos[self.cube_qpos + 3:self.cube_qpos + 7].copy()

        # Random start face keeps the policy general; the palm-center
        # position is always the one authored in scene.xml (never randomized
        # away from it -- only jittered by a fraction of a millimeter).
        start_face = int(np.random.choice(6))
        self.start_face = start_face
        self.target_face = int(np.random.choice([f for f in range(6) if f != start_face]))

        up_q = _q_from_vecs(FACE_NORMALS[start_face], np.array([0.0, 0.0, 1.0]))
        quat = _qmul(up_q, base_quat)
        yaw = np.random.uniform(-c.yaw_jitter_rad, c.yaw_jitter_rad)
        quat = _qmul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), quat)
        quat /= np.linalg.norm(quat)

        target_center = self.nominal_cube_center.copy()
        target_center[:2] += np.random.uniform(-c.pos_jitter_m, c.pos_jitter_m, 2)
        R = _quat_to_mat(quat)
        pos = target_center - R @ CUBE_LOCAL_CENTER

        self.data.qpos[self.cube_qpos:self.cube_qpos + 3] = pos
        self.data.qpos[self.cube_qpos + 3:self.cube_qpos + 7] = quat
        self.data.qvel[self.cube_dof:self.cube_dof + 6] = 0.0

        if self.randomize:
            self.model.geom_friction[:] = self.base_friction * np.random.uniform(0.8, 1.2)

        self.step_count = 0
        self._armed = True
        self._hold = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._last_success_step = -10_000

        mujoco.mj_forward(self.model, self.data)

        # Ramp the grasp shut instead of teleporting to HAND_CLOSE_FRAC:
        # a cube on a random start_face isn't rotationally symmetric to the
        # fingertip geometry, so an instant fully-closed qpos can start
        # already penetrating the mesh -- the contact solver then explodes
        # that in one step. This settle loop is a fail-safe against that.
        qpos_closed = np.clip(
            self.ctrl_mid + self.close_sign * HAND_CLOSE_FRAC * self.ctrl_half
            + np.random.uniform(-HAND_CLOSE_JITTER, HAND_CLOSE_JITTER, N_JOINTS),
            self.ctrl_lo, self.ctrl_hi)
        for i in range(SETTLE_STEPS):
            frac = (i + 1) / SETTLE_STEPS
            self.data.ctrl[self.actids] = qpos_open + frac * (qpos_closed - qpos_open)
            mujoco.mj_step(self.model, self.data)

        qpos_init = self.data.qpos[self.qposids].copy()
        frac_init = np.clip((qpos_init - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        self.grasp_frac[:] = frac_init
        self.filtered_ctrl[:] = frac_init
        self.last_action[:] = 0.0

        self.prev_theta = self._theta()
        self.start_pos = self._cube_center_world().copy()
        return self._get_obs()

    # -- task-relevant scalars --------------------------------------------
    def _cube_rotmat(self):
        return self.data.xmat[self.cubeid].reshape(3, 3)

    def _alignment(self):
        """cos(angle) between the target face's outward normal (rotated into
        world frame) and world +Z -- 1.0 means the target face is on top."""
        n = self._cube_rotmat() @ FACE_NORMALS[self.target_face]
        return float(n[2])

    def _theta(self):
        return float(np.arccos(np.clip(self._alignment(), -1.0, 1.0)))

    def _cube_angvel(self):
        return float(np.linalg.norm(self.data.qvel[self.cube_dof + 3:self.cube_dof + 6]))

    def _tip_to_cube(self):
        cpos = self._cube_center_world()
        return np.array([cpos - self.data.site_xpos[sid] for sid in self.tip_sites], dtype=np.float32)

    def _n_tips_touching(self):
        """Diagnostic only -- never rewarded, so the policy can't farm
        'contact' as a proxy instead of actually rotating the cube."""
        c_force = 0.015
        touched = np.zeros(len(self.tip_sites), dtype=bool)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 not in self.cube_geoms and con.geom2 not in self.cube_geoms:
                continue
            other = con.geom2 if con.geom1 in self.cube_geoms else con.geom1
            for t, geoms in enumerate(self.tip_geoms):
                if other in geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    if abs(f6[0]) > c_force:
                        touched[t] = True
        return int(touched.sum())

    # -- observation ---------------------------------------------------------
    def _get_obs(self):
        jpos = np.clip((self.data.qpos[self.qposids] - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        jvel = np.clip(self.data.qvel[self.qvelids] / JVEL_SCALE, -1.0, 1.0)

        cube_up_local = self._cube_rotmat().T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        onehot = np.zeros(6, np.float32)
        onehot[self.target_face] = 1.0

        obs = np.concatenate([jpos, jvel, self.last_action, cube_up_local, onehot]).astype(np.float32)

        # Fail-safe: a NaN/inf here means the physics step diverged
        # (penetration blow-up, etc.). Sanitize rather than crash training,
        # and let the caller flag the episode as unsafe via `dropped`.
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    # -- the 3-term reward ------------------------------------------------
    def _reward(self, dropped):
        c = self.cfg

        theta = self._theta()
        # Progress term, clipped: without this clip, a single contact-solver
        # glitch that teleports theta can hand the policy a huge one-step
        # reward it didn't earn -- the paper's structure assumes smooth
        # progress, this clip enforces that assumption holds numerically.
        d_theta = float(np.clip(self.prev_theta - theta, -c.rotate_clip_rad, c.rotate_clip_rad))
        r_rotate = c.k_rotate * d_theta
        self.prev_theta = theta

        r_drop = -c.drop_penalty if dropped else 0.0

        angvel = self._cube_angvel()
        settled = theta < c.success_theta_rad and angvel < c.success_max_angvel
        r_success, success = 0.0, False
        steps_since = self.step_count - self._last_success_step
        if self._armed and steps_since >= c.min_steps_between_success:
            # Must stay settled for `success_hold_steps` in a row -- guards
            # against farming the bonus by flickering across the threshold.
            self._hold = self._hold + 1 if settled else 0
            if self._hold >= c.success_hold_steps:
                r_success = c.success_bonus
                success = True
                self._armed = False
                self._hold = 0
                self._last_success_step = self.step_count
                # Never stop the episode on success: immediately hand out a
                # new target so the policy keeps rotating instead of
                # settling into a "reached goal once, now sit still" minimum.
                self.target_face = int(np.random.choice([f for f in range(6) if f != self.target_face]))
                self.prev_theta = self._theta()
        elif not self._armed and theta > c.success_rearm_theta_rad:
            # Must visibly leave the target before it can be re-claimed --
            # closes the same flicker exploit from the other side.
            self._armed = True

        total = r_rotate + r_success + r_drop
        info = {
            "theta_rad": theta, "success": success, "dropped": dropped,
            "target_face": self.target_face, "start_face": self.start_face,
            "n_tips_touching": self._n_tips_touching(),
            "reach_dist": float(np.mean(np.linalg.norm(self._tip_to_cube(), axis=1))),
            "cube_angvel": angvel,
            "r_rotate": r_rotate, "r_success": r_success, "r_drop": r_drop,
        }
        return total, info

    # -- step ---------------------------------------------------------------
    def step(self, action):
        c = self.cfg
        act = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_frac = np.clip(self.grasp_frac + act * c.grasp_band_frac, -1.0, 1.0)
        lpf_ctrl = c.action_lpf * target_frac + (1 - c.action_lpf) * self.filtered_ctrl
        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl, -c.max_ctrl_rate_frac, c.max_ctrl_rate_frac)
        self.filtered_ctrl = self.filtered_ctrl + step_delta

        ctrl = np.clip(self.ctrl_mid + self.filtered_ctrl * self.ctrl_half, self.ctrl_lo, self.ctrl_hi)
        full = np.zeros(self.model.nu)
        full[self.actids] = ctrl
        self.do_simulation(full, self.frame_skip)
        self.step_count += 1

        cpos = self._cube_center_world()
        z_drop_actual = float(self.start_pos[2] - cpos[2])
        xy_drift_actual = float(np.linalg.norm(cpos[:2] - self.start_pos[:2]))

        self._xy_over_steps = self._xy_over_steps + 1 if xy_drift_actual > c.xy_drop_m else 0
        self._z_over_steps = self._z_over_steps + 1 if z_drop_actual > c.z_drop_m else 0
        dropped = (self._xy_over_steps >= c.drop_persist_steps
                   or self._z_over_steps >= c.drop_persist_steps
                   or not np.all(np.isfinite(cpos)))  # fail-safe: divergent sim counts as a drop

        reward, info = self._reward(dropped)
        info["xy_drift_m"] = xy_drift_actual
        info["z_drop_m_actual"] = z_drop_actual
        self.last_action = act.copy()

        done = bool(dropped)  # episode always ends on drop; never on success -- see _reward()
        truncated = self.step_count >= c.max_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, done, truncated, info