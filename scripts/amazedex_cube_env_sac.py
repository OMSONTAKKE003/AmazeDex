import os
from dataclasses import dataclass
import mujoco
import numpy as np
from gymnasium import spaces
from rich.progress import Progress
from mujoco_env import MujocoEnv

# Joints, actuators, and tip sites of the robot hand
ACTUATORS = [
    "motor_finger1_1", "motor_finger1_2", "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2", "motor_finger4_1", "motor_finger4_2"
]
JOINTS = [
    "finger1_motor1", "finger1_motor2", "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2", "finger4_motor1", "finger4_motor2"
]
TIP_SITES = ["tip1", "tip2", "tip3", "tip4"]

MODEL_PATH = os.environ.get(
    "AMAZEDEX_MODEL_PATH",
    r"/home/om/amexdexluv/AmazeDex/AmazeDex/resources/scene.xml",
)

FACE_NORMALS = np.array(
    [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]],
    dtype=np.float32,
)
FACE_NAMES = ["-X", "-Z", "+Y", "-Y", "+X", "+Z"]

ADJACENT_FACES = {f: [g for g in range(6) if g != f and g != (f ^ 1)] for f in range(6)}
OPPOSITE_FACE = {f: f ^ 1 for f in range(6)}

CUBE_LOCAL_CENTER = np.array([-0.0028, -0.035, 0.0011], dtype=np.float32)
N_JOINTS = 8
N_TIPS = 4

JVEL_SCALE = 5.0
ANGVEL_SCALE = 10.0
REACH_NORM = 0.10

HAND_CLOSE_FRAC = 0.50
HAND_CLOSE_JITTER = 0.005
CLOSE_PROBE_FRAC = 0.3
GRASP_OPEN_FRAC = 0.05
SETTLE_STEPS = 30

MAX_CTRL_RATE_FRAC = 0.16

HALF = 0.0235
_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float32,
)
LOCAL_CORNERS = _CORNER_SIGNS * HALF
EDGE_PAIRS = np.array(
    [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.sum(_CORNER_SIGNS[i] != _CORNER_SIGNS[j]) == 1
    ],
    dtype=np.int64,
)
EDGE_TOL = 0.008


@dataclass(frozen=True)
class Cfg:
    max_steps: int = 1000
    z_drop_m: float = 0.18
    xy_drop_m: float = 0.19
    drop_persist_steps: int = 3
    pos_jitter_m: float = 0.000
    yaw_jitter_rad: float = float(np.radians(1.0))

    grasp_band_frac: float = 0.50
    action_lpf: float = 0.42

    max_ctrl_rate_frac: float = MAX_CTRL_RATE_FRAC

    k_align: float = 3.9
    align_clip_rad: float = 0.30
    k_action_rate: float = 0.07

    k_push: float = 0.22
    k_commit: float = 0.090
    push_theta_eps: float = 0.01
    push_commit_steps: int = 2

    k_axis_spin: float = 0.08
    axis_spin_clip: float = 2.0

    k_reach: float = 0.05
    k_edge_bonus: float = 0.15
    reach_target_dist_weight: float = 6.0

    success_theta_rad: float = 0.50
    success_hold_steps: int = 2
    success_max_angvel: float = 4.0
    success_rearm_theta_rad: float = 0.6
    success_bonus: float = 16.2
    min_steps_between_success: int = 2

    success_theta_rad_easy: float = 0.65
    success_hold_steps_easy: int = 1

    drop_penalty: float = 5.1

    terminate_on_success: bool = False


CFG = Cfg()


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _q_from_vecs(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
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


class AmazeDexCubeEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, model_path=MODEL_PATH, render_mode=None, randomize=False, cfg=CFG, **kw):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        super().__init__(model_path, frame_skip=10, render_mode=render_mode)
        self.cfg = cfg
        self.randomize = randomize

        self.actids = np.array([self.model.actuator(n).id for n in ACTUATORS])
        self.qposids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.qvelids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.cubeid = self.model.body("cube").id
        cj = self.model.body("cube").jntadr[0]
        self.cube_qpos = self.model.jnt_qposadr[cj]
        self.cube_dof = self.model.jnt_dofadr[cj]

        self.tip_sites = [self.model.site(n).id for n in TIP_SITES]
        tip_bodies = [self.model.site_bodyid[s] for s in self.tip_sites]
        self.tip_geoms = [
            {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b}
            for b in tip_bodies
        ]
        self.cube_geoms = {
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.cubeid
        }
        self.base_friction = self.model.geom_friction.copy()

        self.nominal_cube_center = (
            self.init_qpos[self.cube_qpos:self.cube_qpos + 3].copy() + CUBE_LOCAL_CENTER
        )

        lims = self.model.actuator_ctrlrange[self.actids]
        self.ctrl_lo, self.ctrl_hi = lims[:, 0], lims[:, 1]
        self.ctrl_mid = (self.ctrl_lo + self.ctrl_hi) / 2
        self.ctrl_half = (self.ctrl_hi - self.ctrl_lo) / 2

        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(55,), dtype=np.float32)

        self.last_action = np.zeros(N_JOINTS, np.float32)
        self.filtered_ctrl = np.zeros(N_JOINTS, np.float32)
        self.grasp_frac = np.zeros(N_JOINTS, np.float32)
        self.step_count = 0
        self.target_face = 1
        self.start_face = 0
        self.start_pos = np.zeros(3)
        self.prev_theta = np.pi
        self.prev_reach = 0.0
        self._prev_tip_target_dist = None
        self._armed = True
        self._hold = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._last_success_step = -10_000
        self._tip_push_streak = np.zeros(N_TIPS, dtype=int)
        self._episode_success = False
        self._episode_success_count = 0
        self.close_sign = self._detect_close_sign()
        self._curriculum_level = 0

    def _cube_center_world(self):
        R = self.data.xmat[self.cubeid].reshape(3, 3)
        return self.data.xpos[self.cubeid] + R @ CUBE_LOCAL_CENTER

    def _detect_close_sign(self):
        qpos_save = self.data.qpos.copy()
        dists = {}
        for sign in (1.0, -1.0):
            self.data.qpos[self.qposids] = self.ctrl_mid + sign * CLOSE_PROBE_FRAC * self.ctrl_half
            mujoco.mj_forward(self.model, self.data)
            cpos = self._cube_center_world()
            dists[sign] = float(
                np.mean([np.linalg.norm(self.data.site_xpos[s] - cpos) for s in self.tip_sites])
            )
        self.data.qpos[:] = qpos_save
        mujoco.mj_forward(self.model, self.data)
        return 1.0 if dists[1.0] < dists[-1.0] else -1.0

    @property
    def targetface(self):
        return self.target_face

    @targetface.setter
    def targetface(self, value):
        self.target_face = value
        self.prev_theta = self._theta()
        self._prev_tip_target_dist = None
        self._armed = True
        self._hold = 0

    def set_curriculum_level(self, level: int) -> None:
        self._curriculum_level = int(level)

    def _sample_target_face(self, exclude: int) -> int:
        if self._curriculum_level == 0:
            candidates = ADJACENT_FACES[exclude]
        else:
            candidates = [f for f in range(6) if f != exclude]
        return int(np.random.choice(candidates))

    def reset_model(self):
        c = self.cfg
        qpos_open = self.ctrl_mid + self.close_sign * GRASP_OPEN_FRAC * self.ctrl_half
        qpos_open = np.clip(qpos_open, self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qposids] = qpos_open
        self.data.qvel[self.qvelids] = 0.0
        self.data.ctrl[self.actids] = qpos_open

        base_quat = self.init_qpos[self.cube_qpos + 3:self.cube_qpos + 7].copy()

        start_face = 0
        self.start_face = start_face
        self.target_face = self._sample_target_face(start_face)
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
        self.drop_counter = 0
        self._tip_push_streak = np.zeros(N_TIPS, dtype=int)
        self._episode_success = False
        self._episode_success_count = 0

        mujoco.mj_forward(self.model, self.data)

        qpos_closed = self.ctrl_mid + self.close_sign * HAND_CLOSE_FRAC * self.ctrl_half
        qpos_closed += np.random.uniform(-HAND_CLOSE_JITTER, HAND_CLOSE_JITTER, N_JOINTS)
        qpos_closed = np.clip(qpos_closed, self.ctrl_lo, self.ctrl_hi)
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
        self.prev_reach = self._reach_dist()
        self._prev_tip_target_dist = None
        self.start_pos = self._cube_center_world().copy()
        return self._get_obs()

    def _alignment(self):
        n = self.data.xmat[self.cubeid].reshape(3, 3) @ FACE_NORMALS[self.target_face]
        return float(n[2])

    def _theta(self):
        return float(np.arccos(np.clip(self._alignment(), -1.0, 1.0)))

    def _cube_angvel(self):
        return float(np.linalg.norm(self.data.qvel[self.cube_dof + 3:self.cube_dof + 6]))

    def _tip_to_cube(self):
        cpos = self._cube_center_world()
        return np.array([cpos - self.data.site_xpos[sid] for sid in self.tip_sites], dtype=np.float32)

    def _reach_dist(self):
        return float(np.min(np.linalg.norm(self._tip_to_cube(), axis=1)))

    def _tips_touching_mask(self):
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
        return touched

    def _n_tips_touching(self):
        return int(self._tips_touching_mask().sum())

    def _cube_frame(self):
        R = self.data.xmat[self.cubeid].reshape(3, 3)
        return self._cube_center_world(), R

    def _world_corners(self, center, R):
        return center + LOCAL_CORNERS @ R.T

    def _tip_targets(self, axis):
        n_tips = len(self.tip_sites)
        center, R = self._cube_frame()
        if axis is None:
            return np.tile(center, (n_tips, 1)).astype(np.float32), np.zeros(n_tips, dtype=np.float32)

        corners = self._world_corners(center, R)
        tip_pos = np.array([self.data.site_xpos[s] for s in self.tip_sites])
        w = self.cfg.reach_target_dist_weight

        r_c = corners - center
        lever_c = np.linalg.norm(np.cross(axis, r_c), axis=1)
        dist_c = np.linalg.norm(tip_pos[:, None, :] - corners[None, :, :], axis=2)
        score_c = lever_c[None, :] - w * dist_c

        a = corners[EDGE_PAIRS[:, 0]]
        b = corners[EDGE_PAIRS[:, 1]]
        ab = b - a
        ab_len2 = np.sum(ab * ab, axis=1) + 1e-12
        diff = tip_pos[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(diff * ab[None, :, :], axis=2) / ab_len2[None, :], 0.0, 1.0)
        cp = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        r_e = cp - center[None, None, :]
        lever_e = np.linalg.norm(np.cross(axis, r_e), axis=2)
        dist_e = np.linalg.norm(tip_pos[:, None, :] - cp, axis=2)
        score_e = lever_e - w * dist_e

        all_scores = np.concatenate([score_c, score_e], axis=1)
        all_points = np.concatenate([np.tile(corners[None, :, :], (n_tips, 1, 1)), cp], axis=1)
        all_lever = np.concatenate([np.tile(lever_c[None, :], (n_tips, 1)), lever_e], axis=1)

        best_idx = np.argmax(all_scores, axis=1)
        rows = np.arange(n_tips)
        best_points = all_points[rows, best_idx]
        best_lever = all_lever[rows, best_idx]

        vec = best_points - center[None, :]
        vec_norm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9
        targets = best_points - EDGE_TOL * vec / vec_norm
        return targets.astype(np.float32), best_lever.astype(np.float32)

    def _reach_shaping(self, axis, touched_mask):
        c = self.cfg
        targets, _ = self._tip_targets(axis)
        tip_pos = np.array([self.data.site_xpos[s] for s in self.tip_sites])
        dists = np.linalg.norm(tip_pos - targets, axis=1)
        if self._prev_tip_target_dist is None:
            self._prev_tip_target_dist = dists.copy()
            return 0.0
        d_dist = self._prev_tip_target_dist - dists
        self._prev_tip_target_dist = dists.copy()
        active = ~touched_mask
        return c.k_reach * float(np.sum(np.clip(d_dist, -REACH_NORM, REACH_NORM) * active))

    def _desired_rot_axis(self):
        n = self.data.xmat[self.cubeid].reshape(3, 3) @ FACE_NORMALS[self.target_face]

        axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
        norm = np.linalg.norm(axis)

        if norm < 1e-6:
            if np.dot(n, np.array([0.0, 0.0, 1.0])) < 0:
                return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            return None

        return axis / norm

    def _point_to_cube_edges_dist(self, p, corners):
        d_corner = float(np.min(np.linalg.norm(corners - p[None, :], axis=1)))
        a = corners[EDGE_PAIRS[:, 0]]
        b = corners[EDGE_PAIRS[:, 1]]
        ab = b - a
        ab_len2 = np.sum(ab * ab, axis=1) + 1e-12
        t = np.clip(np.sum((p[None, :] - a) * ab, axis=1) / ab_len2, 0.0, 1.0)
        cp = a + t[:, None] * ab
        d_edge = float(np.min(np.linalg.norm(cp - p[None, :], axis=1)))
        return min(d_corner, d_edge)

    def _push_alignment(self, axis):
        n_tips = len(self.tip_sites)
        useful = np.zeros(n_tips, dtype=bool)
        near_edge = np.zeros(n_tips, dtype=bool)
        if axis is None:
            return useful, near_edge
        c_force = 0.015
        center, R = self._cube_frame()
        corners = self._world_corners(center, R)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if con.geom1 not in self.cube_geoms and con.geom2 not in self.cube_geoms:
                continue
            other = con.geom2 if con.geom1 in self.cube_geoms else con.geom1
            for t, geoms in enumerate(self.tip_geoms):
                if other not in geoms:
                    continue
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                if abs(f6[0]) <= c_force:
                    continue
                frame = con.frame.reshape(3, 3)
                f_tan_world = f6[1] * frame[1] + f6[2] * frame[2]
                r = con.pos - center
                tau_along = float(np.dot(np.cross(r, f_tan_world), axis))
                if tau_along > 0:
                    useful[t] = True
                    if self._point_to_cube_edges_dist(con.pos, corners) < EDGE_TOL:
                        near_edge[t] = True
        return useful, near_edge

    def _get_obs(self):
        jpos = (self.data.qpos[self.qposids] - self.ctrl_mid) / self.ctrl_half
        jvel = np.clip(self.data.qvel[self.qvelids] / JVEL_SCALE, -1.0, 1.0)
        cubequat = self.data.xquat[self.cubeid].astype(np.float32)
        cube_angvel = np.clip(
            self.data.qvel[self.cube_dof + 3:self.cube_dof + 6] / ANGVEL_SCALE, -1.0, 1.0
        )
        target_normal = self.data.xmat[self.cubeid].reshape(3, 3) @ FACE_NORMALS[self.target_face]
        onehot = np.zeros(6, np.float32)
        onehot[self.target_face] = 1.0

        tip_to_cube = self._tip_to_cube()
        tip_to_cube_n = np.clip(tip_to_cube / REACH_NORM, -3.0, 3.0).flatten()

        axis = self._desired_rot_axis()
        axis_obs = axis.astype(np.float32) if axis is not None else np.zeros(3, np.float32)

        return np.concatenate([
            jpos, jvel, self.last_action, cubequat, cube_angvel,
            target_normal, onehot, tip_to_cube_n, axis_obs
        ]).astype(np.float32)

    def _reward(self, act, dropped):
        c = self.cfg

        theta = self._theta()
        d_theta = float(np.clip(self.prev_theta - theta, -c.align_clip_rad, c.align_clip_rad))
        r_align = c.k_align * d_theta
        self.prev_theta = theta

        r_drop = -c.drop_penalty if dropped else 0.0

        r_action_rate = -c.k_action_rate * float(np.mean(np.square(act - self.last_action)))

        axis = self._desired_rot_axis()
        touched_mask = self._tips_touching_mask()
        useful, near_edge = self._push_alignment(axis)
        outcome_gate = d_theta > c.push_theta_eps
        active_push = useful & outcome_gate

        r_push = c.k_push if active_push.any() else 0.0

        self._tip_push_streak = np.where(active_push, self._tip_push_streak + 1, 0)
        commit_progress = np.clip(self._tip_push_streak / c.push_commit_steps, 0.0, 1.0)
        r_commit = c.k_commit * float(np.max(commit_progress)) if np.any(active_push) else 0.0

        if axis is not None:
            angvel_vec = self.data.qvel[self.cube_dof + 3:self.cube_dof + 6]
            r_axis_spin = c.k_axis_spin * float(
                np.clip(np.dot(angvel_vec, axis), -c.axis_spin_clip, c.axis_spin_clip)
            )
        else:
            r_axis_spin = 0.0

        r_edge_bonus = 0.0

        n_touch = int(touched_mask.sum())

        r_reach = self._reach_shaping(axis, touched_mask)

        reach = self._reach_dist()

        theta_gate = c.success_theta_rad_easy if self._curriculum_level == 0 else c.success_theta_rad
        hold_gate = c.success_hold_steps_easy if self._curriculum_level == 0 else c.success_hold_steps

        angvel = self._cube_angvel()
        settled = theta < theta_gate and angvel < c.success_max_angvel

        r_success, success = 0.0, False
        steps_since = self.step_count - self._last_success_step
        if self._armed and steps_since >= c.min_steps_between_success:
            self._hold = self._hold + 1 if settled else 0
            if self._hold >= hold_gate:
                r_success = c.success_bonus
                success = True
                self._episode_success = True
                self._episode_success_count += 1
                self._armed = False
                self._hold = 0
                self._last_success_step = self.step_count
                if not c.terminate_on_success:
                    self.target_face = self._sample_target_face(self.target_face)
                    self.prev_theta = self._theta()
                    self._prev_tip_target_dist = None
        elif not self._armed and theta > c.success_rearm_theta_rad:
            self._armed = True

        total = (r_align + r_success + r_drop + r_action_rate
                 + r_push + r_commit + r_reach + r_axis_spin)
        info = {
            "theta_rad": theta, "success": success, "dropped": dropped,
            "episode_success": self._episode_success,
            "episode_success_count": self._episode_success_count,
            "target_face": self.target_face, "start_face": self.start_face,
            "n_tips_touching": n_touch,
            "reach_dist": reach, "cube_angvel": angvel,
            "curriculum_level": self._curriculum_level,
            "r_align": r_align, "r_success": r_success, "r_drop": r_drop,
            "r_action_rate": r_action_rate,
            "r_push": r_push, "r_commit": r_commit,
            "r_reach": r_reach,
            "r_edge_bonus": r_edge_bonus,
            "r_axis_spin": r_axis_spin,
            "r_total": total,
        }
        return total, info

    def step(self, action):
        c = self.cfg
        act = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_frac = np.clip(self.grasp_frac + act * c.grasp_band_frac, -1.0, 1.0)
        lpf_ctrl = c.action_lpf * target_frac + (1 - c.action_lpf) * self.filtered_ctrl

        step_delta = np.clip(lpf_ctrl - self.filtered_ctrl, -c.max_ctrl_rate_frac, c.max_ctrl_rate_frac)
        self.filtered_ctrl = self.filtered_ctrl + step_delta

        ctrl = self.ctrl_mid + self.filtered_ctrl * self.ctrl_half
        full = np.zeros(self.model.nu)
        full[self.actids] = ctrl
        self.do_simulation(full, self.frame_skip)
        self.step_count += 1

        cpos = self._cube_center_world()
        z_drop_actual = float(self.start_pos[2] - cpos[2])
        xy_drift_actual = float(np.linalg.norm(cpos[:2] - self.start_pos[:2]))

        self._xy_over_steps = self._xy_over_steps + 1 if xy_drift_actual > c.xy_drop_m else 0
        self._z_over_steps = self._z_over_steps + 1 if z_drop_actual > c.z_drop_m else 0
        dropped_xy = self._xy_over_steps >= c.drop_persist_steps
        dropped_z = self._z_over_steps >= c.drop_persist_steps
        dropped = dropped_z or dropped_xy

        reward, info = self._reward(act, dropped)

        info["dropped_xy"] = dropped_xy
        info["dropped_z"] = dropped_z
        info["xy_drift_m"] = xy_drift_actual
        info["z_drop_m_actual"] = z_drop_actual
        self.last_action = act.copy()

        done = dropped or (c.terminate_on_success and info["success"])
        truncated = self.step_count >= c.max_steps
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, done, truncated, info