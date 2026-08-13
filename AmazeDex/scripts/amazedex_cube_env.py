# imported the headers 
import os
from dataclasses import dataclass
import mujoco
import numpy as np
from gymnasium import spaces
from rich.progress import Progress
from mujoco_env import MujocoEnv

#joints actuatirs and the tip sites of the robot hand
ACTUATORS = ["motor_finger1_1", "motor_finger1_2", "motor_finger2_1", "motor_finger2_2",
             "motor_finger3_1", "motor_finger3_2", "motor_finger4_1", "motor_finger4_2"]
JOINTS = ["finger1_motor1", "finger1_motor2", "finger2_motor1", "finger2_motor2",
          "finger3_motor1", "finger3_motor2", "finger4_motor1", "finger4_motor2"]
TIP_SITES = ["tip1", "tip2", "tip3", "tip4"]

MODEL_PATH = os.environ.get(
    "AMAZEDEX_MODEL_PATH",
    r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\resources\scene.xml",
)
#calculation done using normals as i think they are more reliable 
FACE_NORMALS = np.array([[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.float32)
FACE_NAMES = ["-X", "-Z", "+Y", "-Y", "+X", "+Z"]

#cube was repeatedly off centred so adde some corrections 
CUBE_LOCAL_CENTER = np.array([0.019423,  -0.031,     0.0000254], dtype=np.float32)
N_JOINTS = 8
N_TIPS = 4

#scalea to convert to -1 to 1 range for the observations
JVEL_SCALE = 5.0
ANGVEL_SCALE = 10.0
REACH_NORM = 0.10

#DEFINE GRASP , NOISE 
HAND_CLOSE_FRAC = 0.63
HAND_CLOSE_JITTER = 0.005
#DETECT EHICH DIRECTION TO MOVE FINGER CLOSER OR FARTHER FROM CUBE
CLOSE_PROBE_FRAC = 0.3
GRASP_OPEN_FRAC = 0.05   
SETTLE_STEPS = 30   

#SPEED LIMIT
MAX_CTRL_RATE_FRAC = 0.16


@dataclass(frozen=True)
class Cfg:
    max_steps: int = 400
    #MAX RANGE IN X AND Y DIRECTION 
    z_drop_m: float = 0.18
    xy_drop_m: float = 0.18
    #HOLD FOR THIS LONG 
    drop_persist_steps: int = 5
    pos_jitter_m: float = 0.0001
    yaw_jitter_rad: float = float(np.radians(0.1))

    grasp_band_frac: float = 0.80
    action_lpf: float = 0.6 #ACTIOMN SMOOTHING

    max_ctrl_rate_frac: float = MAX_CTRL_RATE_FRAC

#rewards for alignment and action rate
    k_align: float = 3.9
    align_clip_rad: float = 0.30
    k_action_rate: float = 0.032

    success_theta_rad: float = 0.40
    success_hold_steps: int = 3
    success_max_angvel: float = 4.0
    success_rearm_theta_rad: float = 0.6
    success_bonus: float = 16.0
    min_steps_between_success: int = 0


    drop_penalty: float = 5.8

    terminate_on_success: bool = True


CFG = Cfg()


def _qmul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                      w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, model_path=MODEL_PATH, render_mode=None, randomize=True, cfg=CFG, **kw):
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
        self.tip_geoms = [{g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b}
                           for b in tip_bodies]
        self.cube_geoms = {g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.cubeid}
        self.base_friction = self.model.geom_friction.copy()


        self.nominal_cube_center = (self.init_qpos[self.cube_qpos:self.cube_qpos + 3].copy()
                                     + CUBE_LOCAL_CENTER)

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
        self._armed = True
        self._hold = 0
        self._xy_over_steps = 0
        self._z_over_steps = 0
        self._last_success_step = -10_000
        self.close_sign = self._detect_close_sign()

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
        self.target_face = value
        self.prev_theta = self._theta()
        self._armed = True
        self._hold = 0

    def reset_model(self):
        c = self.cfg
        qpos_open = self.ctrl_mid + self.close_sign * GRASP_OPEN_FRAC * self.ctrl_half
        qpos_open = np.clip(qpos_open, self.ctrl_lo, self.ctrl_hi)
        self.data.qpos[self.qposids] = qpos_open
        self.data.qvel[self.qvelids] = 0.0
        self.data.ctrl[self.actids] = qpos_open

        base_quat = self.init_qpos[self.cube_qpos + 3:self.cube_qpos + 7].copy()

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
        self.drop_counter = 0

        mujoco.mj_forward(self.model, self.data)

        # Ramp the grasp shut under actual actuator dynamics instead of
        # teleporting straight to HAND_CLOSE_FRAC. A cube rotated to a
        # random start_face isn't rotationally symmetric to the fingertip
        # geometry, so an instant fully-closed qpos can start already
        # penetrating the mesh for some orientations -- the contact solver
        # then resolves that in one step, which is the wild spin-out /
        # ejection this loop was added to fix.
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
        return float(np.mean(np.linalg.norm(self._tip_to_cube(), axis=1)))

    def _n_tips_touching(self):
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

    def _get_obs(self):
        jpos = (self.data.qpos[self.qposids] - self.ctrl_mid) / self.ctrl_half
        jvel = np.clip(self.data.qvel[self.qvelids] / JVEL_SCALE, -1.0, 1.0)
        cubequat = self.data.xquat[self.cubeid].astype(np.float32)
        cube_angvel = np.clip(self.data.qvel[self.cube_dof + 3:self.cube_dof + 6] / ANGVEL_SCALE, -1.0, 1.0)
        target_normal = self.data.xmat[self.cubeid].reshape(3, 3) @ FACE_NORMALS[self.target_face]
        onehot = np.zeros(6, np.float32)
        onehot[self.target_face] = 1.0

        tip_to_cube = self._tip_to_cube()
        tip_to_cube_n = np.clip(tip_to_cube / REACH_NORM, -3.0, 3.0).flatten()
        cube_pos_rel = np.clip((self._cube_center_world() - self.start_pos) / REACH_NORM, -3.0, 3.0)

        return np.concatenate([jpos, jvel, self.last_action, cubequat, cube_angvel,
                                target_normal, onehot, tip_to_cube_n, cube_pos_rel]).astype(np.float32)

    def _reward(self, act, dropped):
        c = self.cfg

        theta = self._theta()
        d_theta = float(np.clip(self.prev_theta - theta, -c.align_clip_rad, c.align_clip_rad))
        r_align = c.k_align * d_theta
        self.prev_theta = theta

        if dropped:
            remaining_frac = np.clip((c.max_steps - self.step_count) / c.max_steps, 0.0, 1.0)
            r_drop = -(c.drop_penalty + 12.20 * remaining_frac)
        else:
            r_drop = 0.0

        r_action_rate = -c.k_action_rate * float(np.mean(np.square(act - self.last_action)))

        reach = self._reach_dist()  # diagnostic only, not rewarded
        angvel = self._cube_angvel()
        settled = theta < c.success_theta_rad and angvel < c.success_max_angvel

        r_success, success = 0.0, False
        steps_since = self.step_count - self._last_success_step
        if self._armed and steps_since >= c.min_steps_between_success:
            self._hold = self._hold + 1 if settled else 0
            if self._hold >= c.success_hold_steps:
                r_success = c.success_bonus
                success = True
                self._armed = False
                self._hold = 0
                self._last_success_step = self.step_count
                if not c.terminate_on_success:
                    self.target_face = int(np.random.choice([f for f in range(6) if f != self.target_face]))
                    self.prev_theta = self._theta()
        elif not self._armed and theta > c.success_rearm_theta_rad:
            self._armed = True

        total = r_align + r_success + r_drop + r_action_rate
        info = {
            "theta_rad": theta, "success": success, "dropped": dropped,
            "target_face": self.target_face, "start_face": self.start_face,
            "n_tips_touching": self._n_tips_touching(),
            "reach_dist": reach, "cube_angvel": angvel,
            "r_align": r_align, "r_success": r_success, "r_drop": r_drop,
            "r_action_rate": r_action_rate,
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