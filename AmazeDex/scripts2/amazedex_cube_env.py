from __future__ import annotations

import os
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_env import MujocoEnv

ACTUATORS = [
    "motor_finger1_1", "motor_finger1_2",
    "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2",
    "motor_finger4_1", "motor_finger4_2",
]

JOINTS = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

TIP_SITES = ["tip1", "tip2", "tip3", "tip4"]

FACENORMALS = np.array([
    [0.0, 0.0, 1.0],   # Face 0: Top (+Z)
    [0.0, 0.0, -1.0],  # Face 1: Bottom (-Z)
    [0.0, 1.0, 0.0],   # Face 2: Front (+Y)
    [0.0, -1.0, 0.0],  # Face 3: Back (-Y)
    [1.0, 0.0, 0.0],   # Face 4: Right (+X)
    [-1.0, 0.0, 0.0],  # Face 5: Left (-X)
], dtype=np.float32)

FACENAMES = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6"}

CURRICULUM_STAGES = ["reach", "touch", "grasp", "lift", "rotate"]

DEFAULT_MODEL_PATH = r"C:\Users\luvja\Desktop\updatedwithstand\AmazeDex\resources\scene.xml"
CUBE_HALF_EXTENT = np.array([0.02, 0.02, 0.02], dtype=np.float64)

DEFAULT_ORIENTATION_WEIGHT = 11.0
DEFAULT_SUCCESS_BONUS = 30.0
DEFAULT_SUCCESS_ALIGNMENT_THRESHOLD = 0.97
DEFAULT_SUCCESS_HOLD_STEPS = 12
DEFAULT_DROP_PENALTY = 20.0
DEFAULT_ACTION_RATE_WEIGHT = 0.0007
DEFAULT_JOINT_LIMIT_WEIGHT = 0.1
DEFAULT_JOINT_LIMIT_MARGIN_RAD = 0.05
DEFAULT_DRIFT_WEIGHT = 0.08
DEFAULT_IDLE_WEIGHT = 0.15
DEFAULT_IDLE_SPEED_MARGIN_RAD_S = 0.15


class AmazeDexCubeEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        render_mode: str | None = None,
        orientation_weight: float = DEFAULT_ORIENTATION_WEIGHT,
        success_bonus: float = DEFAULT_SUCCESS_BONUS,
        success_alignment_threshold: float = DEFAULT_SUCCESS_ALIGNMENT_THRESHOLD,
        success_hold_steps: int = DEFAULT_SUCCESS_HOLD_STEPS,
        drop_penalty: float = DEFAULT_DROP_PENALTY,
        action_rate_weight: float = DEFAULT_ACTION_RATE_WEIGHT,
        joint_limit_weight: float = DEFAULT_JOINT_LIMIT_WEIGHT,
        joint_limit_margin_rad: float = DEFAULT_JOINT_LIMIT_MARGIN_RAD,
        drift_weight: float = DEFAULT_DRIFT_WEIGHT,
        idle_weight: float = DEFAULT_IDLE_WEIGHT,
        idle_speed_margin_rad_s: float = DEFAULT_IDLE_SPEED_MARGIN_RAD_S,
        **kwargs,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find MuJoCo model at '{model_path}'.")

        super().__init__(model_path, frame_skip=10, render_mode=render_mode)
        
        self.actids = np.array([self.model.actuator(name).id for name in ACTUATORS])
        self.jointids = np.array([self.model.joint(name).id for name in JOINTS])
        self.qposids = np.array([self.model.joint(name).qposadr[0] for name in JOINTS])
        self.qvelids = np.array([self.model.joint(name).dofadr[0] for name in JOINTS])
        self.cubeid = self.model.body("cube").id

        cube_jnt = self.model.body("cube").jntadr[0]
        self.cube_qposadr = self.model.jnt_qposadr[cube_jnt]
        self.cube_dofadr = self.model.jnt_dofadr[cube_jnt]

        self.tipsiteids = [
            self.model.site(name).id
            for name in TIP_SITES
            if name in [self.model.site(i).name for i in range(self.model.nsite)]
        ]
        self.tipbodyids = np.array([self.model.site_bodyid[s] for s in self.tipsiteids])

        self.jointlimited = self.model.jnt_limited[self.jointids].astype(bool)
        self.jointrange = self.model.jnt_range[self.jointids].copy()
        limits = self.model.actuator_ctrlrange[self.actids]
        self.minctrl, self.maxctrl = limits[:, 0], limits[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(45,), dtype=np.float32)

        self.lastaction = np.zeros(8, dtype=np.float32)
        self.stepcount = 0
        self.startz = 0.0
        self.startpos = np.zeros(3)
        self.startface = 5
        self.targetface = 1
        self.curriculumstage = "reach"
        self.successstreak = 0
        self.prev_alignment = 0.0

        self.orientation_weight = float(orientation_weight)
        self.success_bonus = float(success_bonus)
        self.success_alignment_threshold = float(success_alignment_threshold)
        self.success_hold_steps = int(success_hold_steps)
        self.drop_penalty = float(drop_penalty)
        self.action_rate_weight = float(action_rate_weight)
        self.joint_limit_weight = float(joint_limit_weight)
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.drift_weight = float(drift_weight)
        self.idle_weight = float(idle_weight)
        self.idle_speed_margin_rad_s = float(idle_speed_margin_rad_s)

    def setcurriculumstage(self, stage: str) -> None:
        if stage in CURRICULUM_STAGES:
            self.curriculumstage = stage

    def _sampletargetface(self) -> int:
        if self.curriculumstage == "reach":
            return 1
        elif self.curriculumstage == "touch":
            return 1
        elif self.curriculumstage == "grasp":
            return int(np.random.choice([1, 2]))
        elif self.curriculumstage == "lift":
            return int(np.random.choice([1, 2, 3]))
        else:
            return int(np.random.choice([0, 1, 2, 3, 4, 5]))

    def reset_model(self) -> None:
        self.data.qpos[self.qposids] = 0.0
        self.data.qvel[self.qvelids] = 0.0

        cube_pos = self.init_qpos[self.cube_qposadr : self.cube_qposadr + 3].copy()
        cube_quat = self.init_qpos[self.cube_qposadr + 3 : self.cube_qposadr + 7].copy()

        self.data.qpos[self.cube_qposadr : self.cube_qposadr + 3] = cube_pos
        self.data.qpos[self.cube_qposadr + 3 : self.cube_qposadr + 7] = cube_quat
        self.data.qvel[self.cube_dofadr : self.cube_dofadr + 6] = 0.0

        self.lastaction[:] = 0.0
        self.stepcount = 0
        self.successstreak = 0
        
        self.startface = 5
        self.targetface = self._sampletargetface()

        mujoco.mj_forward(self.model, self.data)

        target_normal = self.data.xmat[self.cubeid].reshape(3, 3) @ FACENORMALS[self.targetface]
        self.prev_alignment = float(target_normal[2])
        pos = self.data.xpos[self.cubeid].copy()
        self.startz = float(pos[2])
        self.startpos = pos.copy()

    def _get_obs(self) -> np.ndarray:
        jpos = self.data.qpos[self.qposids]
        jvel = self.data.qvel[self.qvelids]
        target_world_normal = self.data.xmat[self.cubeid].reshape(3, 3) @ FACENORMALS[self.targetface]
        
        cube_pos = self.data.xpos[self.cubeid]
        tip_pos = np.array([self.data.site_xpos[i] for i in self.tipsiteids])
        relative_tip_pos = (tip_pos - cube_pos).flatten()

        onehot = np.zeros(6, dtype=np.float32)
        onehot[self.targetface] = 1.0

        return np.concatenate([
            jpos, jvel, self.lastaction, target_world_normal, onehot, 
            relative_tip_pos
        ]).astype(np.float32)

    def _get_contacts(self) -> np.ndarray:
        contacts = np.zeros(len(self.tipsiteids), dtype=bool)
        if self.data.ncon == 0: 
            return contacts
        geom_bodyid = self.model.geom_bodyid
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1, b2 = geom_bodyid[con.geom1], geom_bodyid[con.geom2]
            if b1 != self.cubeid and b2 != self.cubeid: 
                continue
            other_body = b2 if b1 == self.cubeid else b1
            hits = np.where(self.tipbodyids == other_body)[0]
            if hits.size: 
                contacts[hits] = True
        return contacts

    def _get_reward(self, act: np.ndarray, dropped: bool) -> tuple[float, dict[str, float]]:
        cube_pos = self.data.xpos[self.cubeid]
        tip_pos = np.array([self.data.site_xpos[i] for i in self.tipsiteids])
        all_distances = np.linalg.norm(tip_pos - cube_pos, axis=1)
        sorted_d = np.sort(all_distances)
        dist = 0.7 * np.mean(all_distances) + 0.3 * np.mean(sorted_d[:2])
        r_reach = 0.9* np.exp(-30.0* dist)

        contacts = self._get_contacts()
        n_contacts = int(np.sum(contacts))
        contact_scales = {0: 0.0, 1: 1.0, 2: 3.2, 3: 3.2, 4: 3.2}
        r_contact = contact_scales.get(n_contacts, 5.0)

        lift = max(0.0, cube_pos[2] - self.startz)
        r_lift = 40.0 * lift if self.curriculumstage in ["lift", "rotate"] else 0.0

        target_normal = self.data.xmat[self.cubeid].reshape(3, 3) @ FACENORMALS[self.targetface]
        alignment = float(target_normal[2])
        progress = max(0.0, alignment - self.prev_alignment)
        self.prev_alignment = alignment

        if self.curriculumstage == "rotate" and n_contacts >= 2:
            r_orientation = self.orientation_weight * alignment + 8.0 * progress
        else:
            r_orientation = 0.0

        # Stage-specific success validation
        stage_success = False
        if self.curriculumstage == "reach":
            stage_success = dist < 0.02
        elif self.curriculumstage == "touch":
            stage_success = n_contacts >= 1
        elif self.curriculumstage == "grasp":
            stage_success = n_contacts >= 2
        elif self.curriculumstage == "lift":
            stage_success = lift > 0.02
        elif self.curriculumstage == "rotate":
            aligned_now = alignment > self.success_alignment_threshold
            if aligned_now and n_contacts >= 2:
                self.successstreak += 1
            else:
                self.successstreak = 0
            stage_success = self.successstreak >= self.success_hold_steps

        r_success = self.success_bonus if stage_success else 0.0

        r_joint, r_action, r_drift, r_idle, r_drop = 0.0, 0.0, 0.0, 0.0, 0.0
        
        if self.curriculumstage in ["grasp", "lift", "rotate"]:
            if np.any(self.jointlimited):
                jpos = self.data.qpos[self.qposids]
                lo, hi = self.jointrange[:, 0], self.jointrange[:, 1]
                m = max(self.joint_limit_margin_rad, 1e-6)
                violation = np.where(self.jointlimited, np.maximum(np.clip(m-(jpos-lo),0.0,m)/m, np.clip(m-(hi-jpos),0.0,m)/m)**2, 0.0)
                r_joint = self.joint_limit_weight * float(np.sum(violation))

            r_action = self.action_rate_weight * float(np.sum(np.square(act - self.lastaction)))
            r_drift = self.drift_weight * float(np.linalg.norm(cube_pos[:2] - self.startpos[:2]))

            if n_contacts >= 2 and not (self.curriculumstage == "rotate" and alignment > self.success_alignment_threshold):
                speed = float(np.mean(np.abs(self.data.qvel[self.qvelids])))
                r_idle = self.idle_weight * np.clip((self.idle_speed_margin_rad_s - speed) / max(self.idle_speed_margin_rad_s, 1e-6), 0.0, 1.0)
            else:
                r_idle = 0.0

            r_drop = self.drop_penalty if dropped else 0.0

        # Integrated r_success across all stages to incentivize crossing the finish line
        if self.curriculumstage == "reach":
            total = r_reach + r_success - r_action
        elif self.curriculumstage == "touch":
            total = r_reach + r_contact + r_success - r_action
        elif self.curriculumstage == "grasp":
            total = r_reach + r_contact + r_success - r_joint - r_action - r_drop
        elif self.curriculumstage == "lift":
            total = r_reach + r_contact + r_lift + r_success - r_joint - r_action - r_drift - r_drop
        else:
            total = r_reach + r_contact + r_lift + r_orientation + r_success - r_joint - r_action - r_drift - r_idle - r_drop

        breakdown = {
            "r_reach": r_reach, "r_contact": r_contact, "n_contacts": n_contacts, 
            "r_lift": r_lift, "r_orientation": r_orientation, "r_success": r_success,
            "r_joint": r_joint, "r_action": r_action, "r_drift": r_drift, 
            "r_idle": r_idle, "r_drop": r_drop, "alignment": alignment, "success": stage_success
        }
        return total, breakdown

    def step(self, action: np.ndarray):
        act = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self.minctrl + (act + 1.0) * 0.5 * (self.maxctrl - self.minctrl)
        fullctrl = np.zeros(self.model.nu)
        fullctrl[self.actids] = ctrl

        self.do_simulation(fullctrl, self.frame_skip)
        self.stepcount += 1

        cubepos = self.data.xpos[self.cubeid]
        dropped = bool((self.startz - cubepos[2]) > 0.06)

        reward, info = self._get_reward(act, dropped)

        done = dropped or info["success"]
        truncated = self.stepcount >= 500
        self.lastaction = act.copy()

        if self.render_mode == "human": 
            self.render()

        info.update({
            "dropped": dropped, 
            "startface": self.startface,
            "targetface": self.targetface, 
            "targetfacename": FACENAMES[self.targetface], 
            "curriculumstage": self.curriculumstage,
            "stage_success": info["success"]
        })
        return self._get_obs(), reward, done, truncated, info