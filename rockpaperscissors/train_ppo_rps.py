from __future__ import annotations
import os
import register_amazedex_rps_env 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

ENV_ID = "AmazeDex/RockPaperScissors-v0"
N_ENVS = 8
TOTAL_TIMESTEPS = 2_000_000
MODEL_DIR = "models"
LOG_DIR = "logs"

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Adding VecMonitor ensures proper logging of dense rewards
    train_env = VecMonitor(make_vec_env(ENV_ID, n_envs=N_ENVS))
    eval_env = VecMonitor(make_vec_env(ENV_ID, n_envs=1))

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=512,      
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_rps_final")
    model.save(final_path)
    print(f"Training Complete. Best model saved to: {MODEL_DIR}/best_model.zip")

if __name__ == "__main__":
    main()