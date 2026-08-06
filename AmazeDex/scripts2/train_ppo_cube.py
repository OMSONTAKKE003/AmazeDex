from __future__ import annotations

import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, SubprocVecEnv

import register_amazedex_env
from amazedex_cube_env import CURRICULUM_STAGES

ENV_ID = "AmazeDex/CubeRotate-v0"
N_ENVS = 16
TOTAL_TIMESTEPS = 2000000

MODEL_DIR = "models"
LOG_DIR = "logs"
CHECKPOINT_DIR = "models/checkpoints"


class SyncVecNormalizeCallback(BaseCallback):
    """Synchronizes observation & reward normalization statistics from train_env to eval_env."""

    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        if self.train_env.ret_rms is not None and self.eval_env.ret_rms is not None:
            self.eval_env.ret_rms = self.train_env.ret_rms
        return True


class SuccessRateCurriculumCallback(BaseCallback):

    def __init__(self, window_size: int = 100, success_threshold: float = 0.80, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self.success_threshold = success_threshold
        self.success_history = deque(maxlen=window_size)
        self.stages = CURRICULUM_STAGES
        self.stage_idx = 0
        self.prev_alignment = 0.0

    def _on_training_start(self) -> None:
        self.training_env.env_method("setcurriculumstage", self.stages[self.stage_idx])

    def _on_step(self) -> bool:
        for infos in self.locals.get("infos", []):
            if "episode" in infos and "success" in infos:
                self.success_history.append(float(infos["success"]))

        if len(self.success_history) >= self.window_size:
            current_success_rate = float(np.mean(self.success_history))

            if current_success_rate >= self.success_threshold and self.stage_idx < len(self.stages) - 1:
                self.stage_idx += 1
                next_stage = self.stages[self.stage_idx]

                if self.verbose > 0:
                    print(
                        f"\n[Adaptive Curriculum] Success Rate ({current_success_rate:.2%}) >= "
                        f"{self.success_threshold:.2%}. Advancing to: '{next_stage}' at step {self.num_timesteps}"
                    )

                self.training_env.env_method("setcurriculumstage", next_stage)
                self.success_history.clear()

        return True


class PolicyPTCheckpointCallback(BaseCallback):
    """Save the PPO policy weights as a PyTorch .pth file."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"ppo_cube_policy_{self.num_timesteps}_steps.pth"
            )

            torch.save(
                self.model.policy.state_dict(),
                path,
            )

            if self.verbose > 0:
                print(f"\nSaved policy weights: {path}")

        return True


def plot_training_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    monitor_path = os.path.join(LOG_DIR, "train_monitor.csv")
    if os.path.exists(monitor_path):
        df = pd.read_csv(monitor_path, skiprows=1)
        df = df.sort_values("t")
        rolling = df["r"].rolling(50, min_periods=1).mean()
        axes[0].plot(df["t"], df["r"], alpha=0.25, label="episode reward")
        axes[0].plot(df["t"], rolling, label="rolling mean (50 ep)")
        axes[0].set_xlabel("wall-clock time (s)")
        axes[0].set_ylabel("episode reward")
        axes[0].set_title("Training reward")
        axes[0].legend()
    else:
        axes[0].set_title("Training reward (no data found)")

    eval_path = os.path.join(LOG_DIR, "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        timesteps = data["timesteps"]
        mean_r = data["results"].mean(axis=1)
        std_r = data["results"].std(axis=1)
        axes[1].plot(timesteps, mean_r, label="mean eval reward")
        axes[1].fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2)
        axes[1].set_xlabel("timesteps")
        axes[1].set_ylabel("eval reward")
        axes[1].set_title("Evaluation reward")
        axes[1].legend()
    else:
        axes[1].set_title("Evaluation reward (no data found)")

    fig.tight_layout()
    out_path = os.path.join(LOG_DIR, "training_curves.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize WandB experiment tracking
    run = wandb.init(
        project="amazedex-cube-rotate",
        config={
            "env_id": ENV_ID,
            "n_envs": N_ENVS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "batch_size": 256,
            "n_steps": 2048,
            "gamma": 0.99,
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # 1. Environment Construction
    train_env = VecMonitor(
        make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv),
        filename=os.path.join(LOG_DIR, "train_monitor.csv"),
    )
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        gamma=0.99,
    )

    eval_env = VecMonitor(make_vec_env(ENV_ID, n_envs=1))
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
    )

    # 2. PPO Agent Configuration
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        device="cuda",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    # 3. Callbacks Assembly
    sync_norm_cb = SyncVecNormalizeCallback(train_env=train_env, eval_env=eval_env)

    curriculum_cb = SuccessRateCurriculumCallback(
        window_size=100,
        success_threshold=0.80,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_cube_ckpt",
        save_vecnormalize=True,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(20_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    pth_checkpoint_cb = PolicyPTCheckpointCallback(
        save_freq=100_000,
        save_path=CHECKPOINT_DIR,
        verbose=1,
    )

    wandb_cb = WandbCallback(
        model_save_path=MODEL_DIR,
        verbose=1,
    )

    callbacks = CallbackList([
        sync_norm_cb,
        curriculum_cb,
        checkpoint_cb,
        pth_checkpoint_cb,
        eval_cb,
        wandb_cb,
    ])

    # 4. Long-Horizon Training Execution
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # 5. Save Final Artifacts
    model.save(os.path.join(MODEL_DIR, "ppo_cube_facetarget_final"))
    train_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"Training complete. Artifacts saved to: {MODEL_DIR}")

    # Plot and upload curves post-training
    plot_training_curves()

    training_curves_path = os.path.join(LOG_DIR, "training_curves.png")
    if os.path.exists(training_curves_path):
        wandb.log({"training_curves": wandb.Image(training_curves_path)})

    run.finish()


if __name__ == "__main__":
    main()