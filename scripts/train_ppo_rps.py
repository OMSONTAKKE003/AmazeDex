
from __future__ import annotations

import os
import torch
import numpy as np
import wandb
import register_amazedex_rps_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CallbackList, 
    EvalCallback, 
    CheckpointCallback  # <-- Imported CheckpointCallback
)
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

ENV_ID = "AmazeDex/CubeRotate-v0"
N_ENVS = 16
TOTAL_TIMESTEPS = 40_000_000
MODEL_DIR = "models"
LOG_DIR = "logs"

WANDB_PROJECT = "amazedex-cube"
WANDB_ENTITY = None 
WANDB_RUN_NAME = None 


class RewardAndGestureEvalCallback(BaseCallback):
    """Evaluates random episodes, logs average rewards, success rates, and total twist angles."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        episode_rewards = []
        episode_cum_twists = []   # To store total twist per episode
        episode_successes = []    # To store if the 5 revolutions were completed

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done_arr, infos = self.eval_env.step(action)
                ep_reward += float(reward[0])
                done = bool(done_arr[0])
                info = infos[0]

            episode_rewards.append(ep_reward)
            
            # The environment tracks `cum_twist` and `success`.
            # When the loop breaks (done=True), `info` contains the final values for the episode.
            final_cum_twist = float(info.get("cum_twist", 0.0))
            final_success = bool(info.get("success", False))
            
            # Convert twist to degrees for easier interpretation (optional)
            episode_cum_twists.append(np.degrees(final_cum_twist))
            episode_successes.append(final_success)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        
        # Calculate averages for wandb
        mean_cum_twist = float(np.mean(episode_cum_twists))
        success_rate = float(np.mean(episode_successes))

        print(f"\n[EVAL Step {self.num_timesteps:07d}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  --> Mean Total Twist: {mean_cum_twist:.1f} degrees")
        print(f"  --> Success Rate: {success_rate * 100:.1f}%")
        print("-" * 50)

        if wandb.run is not None:
            wandb.log({
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
                "eval/mean_total_twist_deg": mean_cum_twist,
                "eval/success_rate": success_rate,
            }, step=self.num_timesteps)

        return True
def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    config = {
        "env_id": ENV_ID,
        "n_envs": N_ENVS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": 256,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "learning_rate": 3e-4,
        "policy": "MlpPolicy",
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=WANDB_RUN_NAME,
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS)
    eval_env = make_vec_env(ENV_ID, n_envs=1)

    model = PPO(
        config["policy"],
        train_env,
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        learning_rate=config["learning_rate"],
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    eval_freq = max(10_000 // N_ENVS, 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=eval_freq,
        n_eval_episodes=20,
        deterministic=True,
    )

    reward_eval_callback = RewardAndGestureEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=20,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(MODEL_DIR, "wandb"),
        model_save_freq=eval_freq * N_ENVS,
        verbose=2,
    )

    # <-- Added Checkpoint logic here
    # Save a checkpoint every 5,000,000 total timesteps
    checkpoint_freq = max(5000000 // N_ENVS, 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="ppo_rps",
    )

    # <-- Added checkpoint_callback to the list
    callback = CallbackList([eval_callback, reward_eval_callback, wandb_callback, checkpoint_callback])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True,
    )

    # 1. Save SB3 zip format
    final_sb3_path = os.path.join(MODEL_DIR, "ppo_rps_final.zip")
    model.save(final_sb3_path)

    # 2. Save pure PyTorch weights (.pth)
    pth_path = os.path.join(MODEL_DIR, "model.pth")
    torch.save(model.policy.state_dict(), pth_path)

    print(f"\nTraining Complete!")
    print(f" Saved SB3 Model: {final_sb3_path}")
    print(f" Saved PyTorch Weights: {pth_path}")

    run.finish()


if __name__ == "__main__":
    main()