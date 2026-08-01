"""Train PPO on AmazeDex/RockPaperScissors-v0 with PyTorch export (.pth) and Evaluation logging."""
from __future__ import annotations

import os
import torch
import numpy as np
import wandb
import register_amazedex_rps_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

ENV_ID = "AmazeDex/RockPaperScissors-v0"
N_ENVS = 16
TOTAL_TIMESTEPS = 2_000_000
MODEL_DIR = "models"
LOG_DIR = "logs"

WANDB_PROJECT = "amazedex-rps"
WANDB_ENTITY = None 
WANDB_RUN_NAME = None 


class RewardAndGestureEvalCallback(BaseCallback):
    """Evaluates 20 random episodes, logs average rewards and success rates per gesture."""

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        episode_rewards = []
        per_gesture_correct: dict[str, list[bool]] = {}

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
            name = info.get("target_gesture_name", "unknown")
            per_gesture_correct.setdefault(name, []).append(bool(info.get("correct_gesture", False)))

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        print(f"\n[EVAL Step {self.num_timesteps:07d}] Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        log_dict = {
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
        }

        for name, results in per_gesture_correct.items():
            rate = float(np.mean(results))
            log_dict[f"eval/success_rate_{name}"] = rate
            print(f"  --> {name.upper()} Success Rate: {rate * 100:.1f}%")

        print("-" * 50)

        if wandb.run is not None:
            wandb.log(log_dict, step=self.num_timesteps)

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
        "ent_coef": 0.01,
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

    callback = CallbackList([eval_callback, reward_eval_callback, wandb_callback])

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