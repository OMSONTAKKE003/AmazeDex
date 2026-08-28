import argparse
import os
from collections import deque

import imageio
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
import sys 
# Using sbx for JAX/Flax GPU compilation
from sbx import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean

# CHANGED: Reverted to SubprocVecEnv to run parallel physics on multiple CPU cores
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.amazedex_cube_env import AmazeDexCubeEnv, MODEL_PATH

MODEL_DIR = "models"
LOG_DIR = "logs"
CLIP_DIR = "logs/clips"

# All frequencies below are in *environment timesteps* (i.e. what wandb/SB3
# call num_timesteps), not callback calls -- the // n_envs division below is
# what converts a timestep target into a callback-call count for a VecEnv.
EVAL_EVERY_STEPS = 20_000
VIDEO_EVERY_STEPS = 50_000     
CKPT_EVERY_STEPS = 20_000     
EARLY_SAFETY_CKPT_STEPS = 20_000  

INFO_KEYWORDS = ("success", "episode_success", "episode_success_count",
                  "dropped", "dropped_xy", "dropped_z", "theta_rad",
                  "cube_angvel", "n_tips_touching", "reach_dist", "xy_drift_m", "z_drop_m_actual",
                  "r_align",  
                  "r_success",  
                  "r_drop",  
                  "r_action_rate",  
                  "r_push", "r_commit",  
                  "r_edge_bonus",  
                  "r_reach",  
                  "r_axis_spin",  
                  "r_total")  

REWARD_KEYS = ("r_align", "r_success", "r_drop", "r_action_rate",
               "r_push", "r_commit", "r_edge_bonus", "r_reach", "r_axis_spin", "r_total")

DIAG_PRINT_EVERY_STEPS = 10_000   
ROLLOUT_LOG_INTERVAL_EPISODES = 4  

SAC_HPARAMS = dict(
    learning_rate=3e-4,
    buffer_size=1_000_000,        # was 500_000 -- more diverse experience now that curriculum-driven diversity is gone
    learning_starts=10_000,      
    batch_size=512,
    tau=0.005,
    gamma=0.9950,
    # Chunked updates to prevent CPU/GPU context-switching overhead
    train_freq=64,               
    gradient_steps=64,           
    ent_coef="auto_0.2",             # was auto_0.1 -- more exploration needed without curriculum bootstrapping
    policy_kwargs=dict(net_arch=[512, 512]),
)


class InferenceClipCallback(BaseCallback):
    def __init__(self, eval_env, save_freq: int, duration_sec: float = 15.0, fps: int = 30):
        super().__init__()
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.n_steps = int(duration_sec * fps)
        self.fps = fps

    def _get_frame(self):
        try:
            return self.eval_env.render()
        except Exception as e:
            print(f"[InferenceClipCallback] render failed, skipping frame: {e}")
            return None

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        try:
            frames = []
            obs = self.eval_env.reset()
            for _ in range(self.n_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = self.eval_env.step(action)
                frame = self._get_frame()
                if frame is not None:
                    frames.append(frame)
                if done[0]:
                    obs = self.eval_env.reset()

            if not frames:
                return True

            os.makedirs(CLIP_DIR, exist_ok=True)
            local_path = os.path.join(CLIP_DIR, f"clip_{self.num_timesteps}.mp4")
            imageio.mimwrite(local_path, frames, fps=self.fps, quality=8)
            print(f"[InferenceClipCallback] saved {local_path} @ {self.num_timesteps} timesteps")

            if wandb.run is not None:
                video_array = np.array(frames).transpose(0, 3, 1, 2)
                wandb.log({"inference_clip": wandb.Video(video_array, fps=self.fps, format="mp4")},
                           step=self.num_timesteps)
        except Exception as e:
            print(f"[InferenceClipCallback] clip capture failed, skipping this clip: {e}")
        return True


class DiagnosticPrintCallback(BaseCallback):
    def __init__(self, eval_every_steps: int = DIAG_PRINT_EVERY_STEPS,
                 rollout_log_interval_episodes: int = ROLLOUT_LOG_INTERVAL_EPISODES):
        super().__init__()
        self.eval_every_steps = eval_every_steps
        self.rollout_log_interval_episodes = rollout_log_interval_episodes
        self._ep_max_xy = None                    
        self._completed_max_xy = deque(maxlen=100)  
        self._episode_count = 0
        self._last_printed_episode_bucket = -1

    def _update_running_max_xy(self) -> None:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None:
            return
        if self._ep_max_xy is None:
            self._ep_max_xy = np.zeros(len(infos))
        for i, info in enumerate(infos):
            xy = info.get("xy_drift_m")
            if xy is not None:
                self._ep_max_xy[i] = max(self._ep_max_xy[i], xy)
            if dones is not None and dones[i]:
                self._completed_max_xy.append(self._ep_max_xy[i])
                self._ep_max_xy[i] = 0.0
                self._episode_count += 1

    def _print_and_record(self) -> None:
        buf = self.model.ep_info_buffer
        if not buf:
            return  

        def mean_of(key: str) -> float:
            vals = [ep[key] for ep in buf if key in ep]
            return float(safe_mean(vals)) if vals else float("nan")

        mean_theta = mean_of("theta_rad")
        mean_xy = mean_of("xy_drift_m")
        mean_z = mean_of("z_drop_m_actual")
        xy_drop_rate = mean_of("dropped_xy")
        z_drop_rate = mean_of("dropped_z")
        success_rate = mean_of("episode_success")
        max_xy = float(np.max(self._completed_max_xy)) if self._completed_max_xy else float("nan")

        reward_means = {key: mean_of(key) for key in REWARD_KEYS}

        print(
            f"[DIAG {self.num_timesteps}] "
            f"mean_theta={mean_theta:.3f} "
            f"mean_xy_drift={mean_xy:.4f} max_xy_drift={max_xy:.4f} "
            f"mean_z_drop={mean_z:.4f} "
            f"xy_drop_rate={xy_drop_rate:.3f} z_drop_rate={z_drop_rate:.3f} "
            f"success_rate={success_rate:.3f} "
            + " ".join(f"{key}={val:.4f}" for key, val in reward_means.items())
        )

        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/drop_rate_xy", xy_drop_rate)
        self.logger.record("rollout/drop_rate_z", z_drop_rate)
        self.logger.record("rollout/theta_rad_mean", mean_theta)
        self.logger.record("rollout/xy_drift_m_mean", mean_xy)
        self.logger.record("rollout/xy_drift_m_max", max_xy)
        self.logger.record("rollout/z_drop_m_mean", mean_z)

        for key, val in reward_means.items():
            self.logger.record(f"reward/{key}", val)

    def _on_step(self) -> bool:
        self._update_running_max_xy()

        n_envs = self.training_env.num_envs
        step_freq_calls = max(self.eval_every_steps // n_envs, 1)
        episode_bucket = self._episode_count // self.rollout_log_interval_episodes

        due_on_eval_cadence = self.n_calls % step_freq_calls == 0
        due_on_rollout_cadence = episode_bucket != self._last_printed_episode_bucket
        if due_on_rollout_cadence:
            self._last_printed_episode_bucket = episode_bucket

        if due_on_eval_cadence or due_on_rollout_cadence:
            self._print_and_record()
        return True


class EarlyCheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, at_step: int = EARLY_SAFETY_CKPT_STEPS):
        super().__init__()
        self.save_path = save_path
        self.at_step = at_step
        self._done = False

    def _on_step(self) -> bool:
        if not self._done and self.num_timesteps >= self.at_step:
            os.makedirs(self.save_path, exist_ok=True)
            model_path = os.path.join(self.save_path, "sac_cube_early_safety")
            buffer_path = os.path.join(self.save_path, "sac_cube_early_safety_buffer")
            self.model.save(model_path)
            self.model.save_replay_buffer(buffer_path)
            print(f"[EarlyCheckpointCallback] safety checkpoint saved @ {self.num_timesteps} timesteps "
                  f"({model_path}.zip)")
            self._done = True
        return True


def _init_wandb_safely(project: str, config: dict) -> bool:
    try:
        wandb.init(project=project, config=config, sync_tensorboard=True,
                   monitor_gym=True, resume="allow")
        return True
    except Exception as e:
        print(f"[wandb] online init failed ({e}), retrying in offline mode")
    try:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=project, config=config, sync_tensorboard=True,
                   monitor_gym=True, resume="allow")
        return True
    except Exception as e:
        print(f"[wandb] offline init also failed ({e}) -- continuing with wandb fully disabled")
        return False


def main(resume_path: str | None, n_envs: int, total_timesteps: int,
         device: str, project: str) -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"MODEL_PATH does not exist: {MODEL_PATH}\n"
            "Set AMAZEDEX_MODEL_PATH to your scene.xml, or fix the default in amazedex_cube_env.py."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    use_wandb = _init_wandb_safely(project, {**SAC_HPARAMS, "n_envs": n_envs,
                                              "total_timesteps": total_timesteps})

    # CHANGED: Replaced DummyVecEnv with SubprocVecEnv to utilize multiprocessing
    train_env = VecMonitor(make_vec_env(AmazeDexCubeEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv,
                                         env_kwargs={"randomize": True}),
                            filename=os.path.join(LOG_DIR, "train_monitor.csv"),
                            info_keywords=INFO_KEYWORDS)
    eval_env = VecMonitor(make_vec_env(AmazeDexCubeEnv, n_envs=1), info_keywords=INFO_KEYWORDS)

    clip_env = VecMonitor(make_vec_env(AmazeDexCubeEnv, n_envs=1, env_kwargs={"render_mode": "rgb_array"}),
                           info_keywords=INFO_KEYWORDS)

    if resume_path and os.path.exists(resume_path):
        model = SAC.load(resume_path, env=train_env, tensorboard_log=LOG_DIR, device=device)
        print(f"Resumed from: {resume_path}")
    else:
        # Dictionary dynamically dictates gradient steps in SAC initialization
        model = SAC("MlpPolicy", train_env, device=device, tensorboard_log=LOG_DIR,
                     verbose=1, **SAC_HPARAMS)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR,
        eval_freq=max(EVAL_EVERY_STEPS // n_envs, 1), n_eval_episodes=20, deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(CKPT_EVERY_STEPS // n_envs, 1),
        save_path=os.path.join(MODEL_DIR, "checkpoints"), name_prefix="sac_cube_ckpt",
     
        save_replay_buffer=True,
        save_vecnormalize=False, 
    )
    clip_cb = InferenceClipCallback(
        clip_env, save_freq=max(VIDEO_EVERY_STEPS // n_envs, 1), duration_sec=15.0, fps=30,
    )
    early_ckpt_cb = EarlyCheckpointCallback(save_path=os.path.join(MODEL_DIR, "checkpoints"))
    diag_cb = DiagnosticPrintCallback(eval_every_steps=DIAG_PRINT_EVERY_STEPS,
                                       rollout_log_interval_episodes=ROLLOUT_LOG_INTERVAL_EPISODES)

    callbacks = [eval_cb, ckpt_cb, clip_cb, early_ckpt_cb, diag_cb]
    if use_wandb:
        callbacks.append(WandbCallback(verbose=2))

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True, reset_num_timesteps=not bool(resume_path),
        log_interval=ROLLOUT_LOG_INTERVAL_EPISODES,
    )

    model.save(os.path.join(MODEL_DIR, "sac_cube_final"))
    print(f"Training complete. Artifacts saved to: {MODEL_DIR}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                         help="SAC reuses samples via replay, so this is a starting point, "
                              "not a floor -- watch eval reward, not this constant.")
    parser.add_argument("--device", type=str, default="cuda", help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--project", type=str, default="amazedex-cube-rotate")
    args = parser.parse_args()
    main(resume_path=args.resume, n_envs=args.n_envs, total_timesteps=args.timesteps,
         device=args.device, project=args.project)