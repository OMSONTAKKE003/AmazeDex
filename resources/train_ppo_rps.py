"""Train PPO on AmazeDex/RockPaperScissors-v0 with Stable-Baselines3.

Uses full state (joint_pos, joint_vel, prev_action, target one-hot) as the
observation during training -- sim2real only needs the *policy*, the real
hand's encoders/servo feedback fill the same slots at inference time.

Usage:
    python train_ppo_rps.py
    tensorboard --logdir logs
    # metrics also stream live to https://wandb.ai/<your-entity>/<project>
"""
from __future__ import annotations

import os

import numpy as np
import wandb
import register_amazedex_rps_env  # noqa: F401  (registers AmazeDex/RockPaperScissors-v0)
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
WANDB_ENTITY = None  # set to your wandb team/username, or leave None to use your default
WANDB_RUN_NAME = None  # None = wandb auto-generates a name


class GestureSuccessCallback(BaseCallback):
    """Logs per-gesture success rate to wandb whenever EvalCallback runs an eval.

    Wraps the eval env with its own rollout so we can bucket results by
    target_gesture_name from info, since EvalCallback itself only reports
    aggregate mean reward.
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        per_gesture_correct: dict[str, list[bool]] = {}
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done_arr, infos = self.eval_env.step(action)
                done = bool(done_arr[0])
                info = infos[0]
            name = info.get("target_gesture_name", "unknown")
            per_gesture_correct.setdefault(name, []).append(bool(info.get("correct_gesture", False)))

        log_dict = {}
        for name, results in per_gesture_correct.items():
            log_dict[f"eval/success_rate_{name}"] = float(np.mean(results))
        if log_dict:
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
        sync_tensorboard=True,  # pulls in everything SB3 already writes to LOG_DIR
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

    gesture_callback = GestureSuccessCallback(
        eval_env=eval_env,
        eval_freq=eval_freq * N_ENVS,  # num_timesteps counts across all envs
        n_eval_episodes=20,
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(MODEL_DIR, "wandb"),
        model_save_freq=eval_freq * N_ENVS,
        verbose=2,
    )

    callback = CallbackList([eval_callback, gesture_callback, wandb_callback])

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_rps_final")
    model.save(final_path)
    print(f"training done. best model: {MODEL_DIR}/best_model.zip, final model: {final_path}.zip")

    run.finish()


if __name__ == "__main__":
    main()