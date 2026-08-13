from __future__ import annotations

import os

import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_env import MujocoEnv

# ==============================================================================
# 0. HAND / TASK CONSTANTS
# ==============================================================================

ACTUATOR_NAMES = [
    "motor_finger1_1", "motor_finger1_2",
    "motor_finger2_1", "motor_finger2_2",
    "motor_finger3_1", "motor_finger3_2",
    "motor_finger4_1", "motor_finger4_2",
]

JOINT_NAMES = [
    "finger1_motor1", "finger1_motor2",
    "finger2_motor1", "finger2_motor2",
    "finger3_motor1", "finger3_motor2",
    "finger4_motor1", "finger4_motor2",
]

TIP_SITE_NAMES = ["tip1", "tip2", "tip3", "tip4"]
CUBE_BODY_NAME = "cube"

# The 6 axis-aligned "face up" target orientations, in MuJoCo's [w, x, y, z]
# convention. Each represents the cube resting with one face pointing +Z
# (world up) after being rotated from the default +Z-up pose.
FACE_UP_QUATS = np.array([
    [1.0,     0.0,     0.0,     0.0],     # +Z face up (default / identity)
    [0.0,     1.0,     0.0,     0.0],     # -Z face up (180° roll)
    [0.7071,  0.0,     0.7071,  0.0],     # +X face up (90° pitch)
    [0.7071,  0.0,    -0.7071,  0.0],     # -X face up (-90° pitch)
    [0.7071, -0.7071,  0.0,     0.0],     # +Y face up (-90° roll)
    [0.7071,  0.7071,  0.0,     0.0],     # -Y face up (90° roll)
], dtype=np.float64)

# Index 0 (+Z / default) is the cube's starting pose, so it's excluded from
# the sampling pool -- there's no reorientation task if target == start.
# Index 1 (-Z / upside-down) is a full 180° flip from default -- hardest.
# Indices 2-5 (+X/-X/+Y/-Y) are 90° pitch/roll away from default -- easiest
# non-trivial targets, since they require only a single quarter-turn.
FACE_EASY_INDICES = np.array([2, 3, 4, 5])   # 90° from default
FACE_HARD_INDICES = np.array([1])            # 180° from default
FACE_ALL_INDICES = np.concatenate([FACE_EASY_INDICES, FACE_HARD_INDICES])

FRAME_SKIP = 1
MAX_STEPS = 800

DROP_HEIGHT = 0.02
LIFT_HEIGHT = 0.04

# --- Reward scales (these are on top of the curriculum-driven weights) ---
LIFT_BONUS_SCALE = 2.0
DROP_PENALTY = 5.0
ACTION_SMOOTHNESS_SCALE = 0.01

# Episode counts as a "success" once the cube is lifted AND its orientation
# reward crosses this threshold (i.e. close enough to the target quaternion).
SUCCESS_ORI_THRESHOLD = 0.95


# ==============================================================================
# 1. CURRICULUM MANAGER
# ==============================================================================

class RewardCurriculumManager:
    """
    Manages training progress and smoothly shifts reward weights, margins,
    and decay temperatures as training progresses from Step 0 to total_training_steps.
    """
    def __init__(self, total_training_steps: int = 40_000_000):
        self.total_steps = total_training_steps
        self.current_step = 0

        # --- 1. Grasping Distance Margin ---
        # Starts forgiving (15 cm margin) -> shrinks to strict (1 cm margin)
        self.grasp_margin_start = 0.15
        self.grasp_margin_end = 0.01

        # --- 2. Dynamic Task Weights (Priority Shift) ---
        # Grasping priority starts high (3.0) -> drops to baseline (1.0)
        self.grasp_weight_start = 3.0
        self.grasp_weight_end = 1.0

        # Orientation priority starts off (0.0) -> becomes the primary goal (3.0)
        self.ori_weight_start = 0.0
        self.ori_weight_end = 3.0

        # --- 3. Orientation Decay Temperature ---
        # Temperature starts at 0.5 (wide/forgiving) -> ramps to 5.0 (sharp/strict)
        self.temp_start = 0.5
        self.temp_end = 5.0

        # --- 4. Penalty Weights ---
        # Hand pose penalty starts at 0.0 (let agent explore) -> ramps to 0.1 (enforce clean pose)
        self.pose_weight_start = 0.0
        self.pose_weight_end = 0.1

        # Forbidden pose penalty starts at 0.5 -> ramps to 2.0 (heavy punishment)
        self.forbidden_weight_start = 0.5
        self.forbidden_weight_end = 2.0

        # --- 5. Phase Boundaries (fractions of total_training_steps) ---
        # Phase 1 [0, phase_lift_start): grasp only, ori_weight pinned at 0,
        #   no target face sampled (agent just learns to approach/grip).
        # Phase 2 [phase_lift_start, phase_reorient_start): grasp margin
        #   keeps tightening, lift bonus already rewards getting the cube
        #   off the table; ori_weight still pinned at 0.
        # Phase 3 [phase_reorient_start, 1.0]: ori_weight ramps 0 -> end,
        #   easy (90°) target faces unlock at phase_reorient_start, hard
        #   (180°) faces unlock at phase_hard_face_start.
        self.phase_lift_start = 0.15
        self.phase_reorient_start = 0.40
        self.phase_hard_face_start = 0.70

    def step_clock(self):
        """Advances the training clock by 1 step (local fallback)."""
        self.current_step += 1

    def set_step(self, step: int):
        """Directly sets the training clock. Used to sync with the true
        global timestep count from an external callback (see
        CurriculumSyncCallback in train_ppo_rps.py), which matters a lot
        once you're running N_ENVS parallel workers -- each worker only
        sees its own local steps, so step_clock() alone would make the
        curriculum progress N_ENVS times slower than reality."""
        self.current_step = max(0, int(step))

    def _get_progress(self) -> float:
        """Returns normalized progress from 0.0 to 1.0."""
        return min(1.0, self.current_step / max(1, self.total_steps))

    def get_grasp_margin(self) -> float:
        # Margin finishes tightening by the start of phase 3, so grasp
        # difficulty doesn't keep increasing on top of the new reorient task.
        p = min(self._get_progress(), self.phase_reorient_start) / self.phase_reorient_start
        p = min(1.0, p)
        return self.grasp_margin_start + p * (self.grasp_margin_end - self.grasp_margin_start)

    def get_grasp_weight(self) -> float:
        p = self._get_progress()
        return self.grasp_weight_start + p * (self.grasp_weight_end - self.grasp_weight_start)

    def get_ori_weight(self) -> float:
        p = self._get_progress()
        if p < self.phase_reorient_start:
            return 0.0
        # Ramp ori_weight over the remaining fraction of training once
        # phase 3 begins, rather than over the whole run.
        adjusted_p = (p - self.phase_reorient_start) / (1.0 - self.phase_reorient_start)
        return self.ori_weight_start + adjusted_p * (self.ori_weight_end - self.ori_weight_start)

    def get_face_sampling_pool(self) -> np.ndarray:
        """No target face is meaningfully sampled until phase 3 begins --
        callers should check is_reorient_phase_active() first and hold
        target_quat at the default pose otherwise."""
        p = self._get_progress()
        if p < self.phase_hard_face_start:
            return FACE_EASY_INDICES
        return FACE_ALL_INDICES

    def is_reorient_phase_active(self) -> bool:
        return self._get_progress() >= self.phase_reorient_start

    def get_current_temperature(self) -> float:
        p = self._get_progress()
        return self.temp_start + p * (self.temp_end - self.temp_start)

    def get_pose_weight(self) -> float:
        p = self._get_progress()
        # Delay hand pose penalty until 20% of training is complete so it learns to grab first
        if p < 0.2:
            return 0.0
        adjusted_p = (p - 0.2) / 0.8
        return self.pose_weight_start + adjusted_p * (self.pose_weight_end - self.pose_weight_start)

    def get_forbidden_weight(self) -> float:
        p = self._get_progress()
        return self.forbidden_weight_start + p * (self.forbidden_weight_end - self.forbidden_weight_start)


# ==============================================================================
# 2. REWARD AND PENALTY FUNCTIONS
# ==============================================================================

def calculate_reach_grasp_reward(data: mujoco.MjData, current_margin: float) -> float:
    """Rewards the agent for bringing its 4 fingertips near the cube."""
    cube_pos = data.body(CUBE_BODY_NAME).xpos
    tip_positions = [data.site(name).xpos for name in TIP_SITE_NAMES]

    distances = [np.linalg.norm(pos - cube_pos) for pos in tip_positions]
    avg_distance = float(np.mean(distances))

    perfect_bound = 0.02  # Within 2 cm of cube center is a perfect 1.0
    if avg_distance <= perfect_bound:
        return 1.0

    distance_outside = avg_distance - perfect_bound
    if distance_outside >= current_margin:
        return 0.0

    reward = 1.0 - (distance_outside / current_margin)
    return float(reward)


def check_cube_lift_reward(data: mujoco.MjData, lift_height: float = LIFT_HEIGHT) -> float:
    """Returns 1.0 if the cube's global Z-position is lifted above lift_height, else 0.0."""
    cube_z_height = data.body(CUBE_BODY_NAME).xpos[2]
    return 1.0 if cube_z_height > lift_height else 0.0


def calculate_dense_orientation_reward(data: mujoco.MjData, target_quat: np.ndarray, temperature: float) -> float:
    """Calculates a dense orientation alignment reward using exponential decay."""
    current_quat = data.body(CUBE_BODY_NAME).xquat
    dot_product = np.dot(current_quat, target_quat)
    abs_dot = np.clip(np.abs(dot_product), -1.0, 1.0)
    theta = 2.0 * np.arccos(abs_dot)
    return float(np.exp(-temperature * theta))


def calculate_hand_pose_penalty(data: mujoco.MjData) -> float:
    """Penalizes deviations of joints from their 0.0 resting positions."""
    default_poses = np.zeros(len(JOINT_NAMES))
    hand_pose_penalty = 0.0

    for i, joint_name in enumerate(JOINT_NAMES):
        current_angle = data.joint(joint_name).qpos[0]
        diff = current_angle - default_poses[i]
        hand_pose_penalty += (diff * diff)

    return float(hand_pose_penalty)


def calculate_forbidden_pose_penalty(data: mujoco.MjData) -> float:
    """Penalizes the awkward backward-bending pose (q1 < -1.0 AND q2 > 1.0)."""
    forbidden_pose_penalty = 0.0
    for i in range(1, 5):
        motor1_name = f"finger{i}_motor1"
        motor2_name = f"finger{i}_motor2"
        q1 = data.joint(motor1_name).qpos[0]
        q2 = data.joint(motor2_name).qpos[0]

        bad_q1_depth = max(0.0, -1.0 - q1)
        bad_q2_depth = max(0.0, q2 - 1.0)

        forbidden_pose_penalty += (bad_q1_depth * bad_q2_depth)

    return float(forbidden_pose_penalty)


def check_drop_condition(data: mujoco.MjData, drop_height: float = DROP_HEIGHT) -> bool:
    """Returns True if the cube drops below the threshold height."""
    cube_z_height = data.body(CUBE_BODY_NAME).xpos[2]
    return bool(cube_z_height < drop_height)


# ==============================================================================
# 3. GYMNASIUM ENVIRONMENT
# ==============================================================================

class AmazeDexCubeGraspEnv(MujocoEnv):
    """
    Grasp -> lift -> orient task for the AmazeDex hand.

    Wraps the reward pieces above behind a curriculum: early in training the
    agent is only asked to get its fingertips near the cube (wide margin,
    orientation weight ~0). As training progresses the margin tightens, the
    orientation term ramps up and sharpens, and pose penalties kick in to
    clean up the grasp -- all driven by RewardCurriculumManager.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(
        self,
        model_path: str = os.path.join("resources", "scene.xml"),
        render_mode: str | None = None,
        total_training_steps: int = 40_000_000,
        randomize_target: bool = True,
        max_steps: int = MAX_STEPS,
    ):
        super().__init__(model_path, FRAME_SKIP, render_mode)

        self.actuator_ids = np.array([self.model.actuator(n).id for n in ACTUATOR_NAMES])
        self.joint_qpos_adr = np.array([self.model.joint(n).qposadr[0] for n in JOINT_NAMES])
        self.joint_dof_adr = np.array([self.model.joint(n).dofadr[0] for n in JOINT_NAMES])
        self.cube_body_id = self.model.body(CUBE_BODY_NAME).id

        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.ctrl_low, self.ctrl_high = ctrl_range[:, 0], ctrl_range[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        # joint pos(8) + joint vel(8) + cube pos(3) + cube quat(4)
        # + target quat(4) + fingertip distances(4) + prev action(8) = 39
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(39,), dtype=np.float32)

        self.curriculum = RewardCurriculumManager(total_training_steps)
        self.randomize_target = randomize_target
        self.max_steps = max_steps

        self.prev_action = np.zeros(8, dtype=np.float32)
        self.steps = 0
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.target_face_idx = 0

    def set_curriculum_step(self, step: int):
        """Called externally (e.g. from a callback) to sync curriculum
        progress with the true global training timestep count."""
        self.curriculum.set_step(step)

    def reset_model(self) -> None:
        self.data.qpos[self.joint_qpos_adr] = 0.0
        self.data.qvel[self.joint_dof_adr] = 0.0
        self.prev_action[:] = 0.0
        self.steps = 0

        if self.randomize_target and self.curriculum.is_reorient_phase_active():
            pool = self.curriculum.get_face_sampling_pool()
            face_idx = int(self.np_random.choice(pool))
            self.target_quat = FACE_UP_QUATS[face_idx].copy()
            self.target_face_idx = face_idx
        else:
            self.target_quat = FACE_UP_QUATS[0].copy()
            self.target_face_idx = 0

    def _fingertip_distances(self) -> np.ndarray:
        cube_pos = self.data.body(CUBE_BODY_NAME).xpos
        return np.array(
            [np.linalg.norm(self.data.site(name).xpos - cube_pos) for name in TIP_SITE_NAMES],
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        joint_pos = self.data.qpos[self.joint_qpos_adr]
        joint_vel = self.data.qvel[self.joint_dof_adr]
        cube_pos = self.data.body(CUBE_BODY_NAME).xpos.copy()
        cube_quat = self.data.body(CUBE_BODY_NAME).xquat.copy()
        tip_dists = self._fingertip_distances()

        return np.concatenate([
            joint_pos, joint_vel,
            cube_pos, cube_quat,
            self.target_quat, tip_dists,
            self.prev_action,
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ctrl = self.ctrl_low + (action + 1.0) * 0.5 * (self.ctrl_high - self.ctrl_low)

        full_ctrl = np.zeros(self.model.nu)
        full_ctrl[self.actuator_ids] = ctrl
        self.do_simulation(full_ctrl, self.frame_skip)
        self.steps += 1

        # Fallback local clock -- overridden by set_curriculum_step() if a
        # training callback is syncing global progress (see train_ppo_rps.py).
        self.curriculum.step_clock()

        # ---- REWARD COMPONENTS ----
        margin = self.curriculum.get_grasp_margin()
        grasp_reward = calculate_reach_grasp_reward(self.data, margin)

        lift_reward = check_cube_lift_reward(self.data)

        temperature = self.curriculum.get_current_temperature()
        ori_reward = calculate_dense_orientation_reward(self.data, self.target_quat, temperature)

        pose_penalty = calculate_hand_pose_penalty(self.data)
        forbidden_penalty = calculate_forbidden_pose_penalty(self.data)
        dropped = check_drop_condition(self.data)

        grasp_w = self.curriculum.get_grasp_weight()
        ori_w = self.curriculum.get_ori_weight()
        pose_w = self.curriculum.get_pose_weight()
        forbidden_w = self.curriculum.get_forbidden_weight()

        action_delta = action - self.prev_action
        smoothness_penalty = -ACTION_SMOOTHNESS_SCALE * float(np.sum(np.square(action_delta)))

        reward = (
            grasp_w * grasp_reward
            + LIFT_BONUS_SCALE * lift_reward
            + ori_w * ori_reward
            - pose_w * pose_penalty
            - forbidden_w * forbidden_penalty
            + smoothness_penalty
        )
        if dropped:
            reward -= DROP_PENALTY

        success = bool(lift_reward > 0.0 and ori_reward >= SUCCESS_ORI_THRESHOLD)
        terminated = bool(dropped or success)
        truncated = self.steps >= self.max_steps

        self.prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        progress = self.curriculum._get_progress()
        if progress < self.curriculum.phase_lift_start:
            phase_name = "grasp"
        elif progress < self.curriculum.phase_reorient_start:
            phase_name = "grasp_lift"
        elif progress < self.curriculum.phase_hard_face_start:
            phase_name = "reorient_easy"
        else:
            phase_name = "reorient_hard"

        info = {
            "grasp_reward": grasp_reward,
            "lift_reward": lift_reward,
            "ori_reward": ori_reward,
            "pose_penalty": pose_penalty,
            "forbidden_penalty": forbidden_penalty,
            "smoothness_penalty": smoothness_penalty,
            "dropped": dropped,
            "success": success,
            "is_success": success,
            "curriculum_progress": progress,
            "grasp_margin": margin,
            "grasp_weight": grasp_w,
            "ori_weight": ori_w,
            "temperature": temperature,
            "target_face_idx": self.target_face_idx,
            "training_phase": phase_name,
        }

        return self._get_obs(), reward, terminated, truncated, info


if __name__ == "__main__":
    env = AmazeDexCubeGraspEnv()
    for episode in range(5):
        obs, info = env.reset(seed=episode)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        reason = "dropped" if info["dropped"] else ("success" if info["success"] else "max steps reached")
        print(
            f"episode {episode}: return={total_reward:.2f}, steps={env.steps}, "
            f"success={info['success']}, ended: {reason}"
        )
    env.close()