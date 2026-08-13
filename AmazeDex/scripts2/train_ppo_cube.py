
import argparse
import os
from collections import deque

import imageio
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from amazedex_cube_env import AmazeDexCubeEnv, MODEL_PATH

MODEL_DIR = "models"
LOG_DIR = "logs"
CLIP_DIR = "logs/clips"

# All frequencies below are in *environment timesteps* (i.e. what wandb/SB3
# call num_timesteps), not callback calls -- the // n_envs division below is
# what converts a timestep target into a callback-call count for a VecEnv.
EVAL_EVERY_STEPS = 50_000
VIDEO_EVERY_STEPS = 50_000
CKPT_EVERY_STEPS = 50_000
DIAG_EVERY_STEPS = 10_000

# These flow straight from the env's info dict into ep_info_buffer, which is
# how we watch for reward hacking without touching the reward itself.
INFO_KEYWORDS = ("success", "dropped", "theta_rad", "cube_angvel",
                  "n_tips_touching", "reach_dist", "xy_drift_m", "z_drop_m_actual")

PPO_HPARAMS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=4096,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
)


class InferenceClipCallback(BaseCallback):
    """Runs a deterministic eval rollout and saves an mp4 every save_freq
    calls. save_freq is in *callback calls*, i.e. already n_envs-adjusted by
    the caller -- see how it's constructed in main()."""

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


class DiagnosticCallback(BaseCallback):
    """Prints/logs every reward-hacking-relevant term from the rolling
    episode buffer every DIAG_EVERY_STEPS timesteps -- see module docstring
    for why success_rate, drop_rate, and mean_theta are the ones to watch,
    but all INFO_KEYWORDS terms are printed so nothing drifts unnoticed."""

    def __init__(self, every_steps: int = DIAG_EVERY_STEPS):
        super().__init__()
        self.every_steps = every_steps
        self._last_step = 0
        self._ep_max_xy = None
        self._completed_max_xy = deque(maxlen=100)  # matches ep_info_buffer's 100-episode window

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

    def _on_step(self) -> bool:
        self._update_running_max_xy()

        if self.num_timesteps - self._last_step < self.every_steps:
            return True
        self._last_step = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf:
            return True

        def mean_of(key):
            vals = [ep[key] for ep in buf if key in ep]
            return float(safe_mean(vals)) if vals else float("nan")

        success_rate = mean_of("success")
        drop_rate = mean_of("dropped")
        mean_theta = mean_of("theta_rad")
        mean_cube_angvel = mean_of("cube_angvel")
        mean_n_tips_touching = mean_of("n_tips_touching")
        mean_reach_dist = mean_of("reach_dist")
        mean_xy_drift = mean_of("xy_drift_m")
        mean_z_drop = mean_of("z_drop_m_actual")
        max_xy_drift = float(np.max(self._completed_max_xy)) if self._completed_max_xy else float("nan")

        print(
            f"[DIAG {self.num_timesteps}] "
            f"success_rate={success_rate:.3f} "
            f"drop_rate={drop_rate:.3f} "
            f"mean_theta={mean_theta:.3f} "
            f"cube_angvel={mean_cube_angvel:.3f} "
            f"n_tips_touching={mean_n_tips_touching:.3f} "
            f"reach_dist={mean_reach_dist:.4f} "
            f"mean_xy_drift={mean_xy_drift:.4f} max_xy_drift={max_xy_drift:.4f} "
            f"mean_z_drop={mean_z_drop:.4f}"
        )

        # Also feed wandb/tensorboard so the same numbers show up as curves,
        # not just console text -- redundant with the print above by design.
        self.logger.record("rollout/success_rate", success_rate)
        self.logger.record("rollout/drop_rate", drop_rate)
        self.logger.record("rollout/theta_rad_mean", mean_theta)
        self.logger.record("rollout/cube_angvel_mean", mean_cube_angvel)
        self.logger.record("rollout/n_tips_touching_mean", mean_n_tips_touching)
        self.logger.record("rollout/reach_dist_mean", mean_reach_dist)
        self.logger.record("rollout/xy_drift_m_mean", mean_xy_drift)
        self.logger.record("rollout/xy_drift_m_max", max_xy_drift)
        self.logger.record("rollout/z_drop_m_mean", mean_z_drop)
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
            "Set AMAZEDEX_MODEL_PATH to your scene.xml before training."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    use_wandb = _init_wandb_safely(project, {**PPO_HPARAMS, "n_envs": n_envs,
                                              "total_timesteps": total_timesteps})

    train_env = VecMonitor(
        make_vec_env(AmazeDexCubeEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv),
        filename=os.path.join(LOG_DIR, "train_monitor.csv"),
        info_keywords=INFO_KEYWORDS,
    )
    eval_env = VecMonitor(make_vec_env(AmazeDexCubeEnv, n_envs=1), info_keywords=INFO_KEYWORDS)

    # Render mode must be passed in via env_kwargs, otherwise the environment
    # defaults to None and generates no array data to save into a clip.
    clip_env = VecMonitor(make_vec_env(AmazeDexCubeEnv, n_envs=1, env_kwargs={"render_mode": "rgb_array"}),
                           info_keywords=INFO_KEYWORDS)

    if resume_path and os.path.exists(resume_path):
        model = PPO.load(resume_path, env=train_env, tensorboard_log=LOG_DIR, device=device)
        print(f"Resumed from: {resume_path}")
    else:
        model = PPO("MlpPolicy", train_env, device=device, tensorboard_log=LOG_DIR,
                     verbose=1, **PPO_HPARAMS)

    eval_cb = EvalCallback(eval_env, best_model_save_path=MODEL_DIR, log_path=LOG_DIR,
                            eval_freq=max(EVAL_EVERY_STEPS // n_envs, 1),
                            n_eval_episodes=20, deterministic=True)
    ckpt_cb = CheckpointCallback(save_freq=max(CKPT_EVERY_STEPS // n_envs, 1),
                                  save_path=os.path.join(MODEL_DIR, "checkpoints"),
                                  name_prefix="ppo_cube_ckpt")
    clip_cb = InferenceClipCallback(clip_env, save_freq=max(VIDEO_EVERY_STEPS // n_envs, 1),
                                     duration_sec=15.0, fps=30)
    diag_cb = DiagnosticCallback(every_steps=DIAG_EVERY_STEPS)

    callbacks = [eval_cb, ckpt_cb, clip_cb, diag_cb]
    if use_wandb:
        callbacks.append(WandbCallback(verbose=2))

    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                progress_bar=True, reset_num_timesteps=not bool(resume_path))

    model.save(os.path.join(MODEL_DIR, "ppo_cube_final"))
    print(f"Training complete. Artifacts saved to: {MODEL_DIR}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=50_000_000,
                         help="Dexterous reorientation from scratch typically needs O(1e7-1e8) "
                              "env steps; watch success_rate in the [DIAG] prints rather than "
                              "treating this as a hard target.")
    parser.add_argument("--device", type=str, default="cuda", help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--project", type=str, default="amazedex-cube-rotate")
    args = parser.parse_args()
    main(resume_path=args.resume, n_envs=args.n_envs, total_timesteps=args.timesteps,
         device=args.device, project=args.project)